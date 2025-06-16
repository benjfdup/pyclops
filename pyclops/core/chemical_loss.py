from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, ClassVar, Union, final
import torch
import mdtraj as md
import parmed as pmd
import warnings
import itertools
import tempfile
import os

from ..torchkde import KernelDensity
from ..utils.indexing import IndexesMethodPair
from ..utils.constants import KB, AMBER_CAPS

class KDEManager:
    """Singleton manager for KDE models to prevent redundant loading"""
    
    _instance = None
    _kde_cache: Dict[str, KernelDensity] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KDEManager, cls).__new__(cls)
        return cls._instance
    
    def get_kde(self, kde_file: str, device: torch.device) -> KernelDensity:
        """Get or load a KDE model from cache using the provided file path."""
        if kde_file in self._kde_cache:
            return self._kde_cache[kde_file]
        
        self._kde_cache[kde_file] = torch.load(kde_file, map_location=device)
        return self._kde_cache[kde_file]
    
    def clear_cache(self) -> None:
        """Clear the KDE model cache to free memory"""
        self._kde_cache.clear()

class ChemicalLoss(ABC):
    """
    ChemicalLoss optimized for torch.jit compilation.
    Instances of this class should be treated as immutable after initialization.
    """
    
    # Class variables to be overridden by subclasses
    atom_idxs_keys: ClassVar[List[str]] = []
    kde_file: ClassVar[str] = ''
    linkage_pdb_file: ClassVar[str] = ''
    
    def __init__(
        self,
        method: str,
        atom_idxs: Dict[str, int],
        temp: float = 1.0,
        weight: float = 1.0,
        offset: float = 0.0,
        resonance_key: Optional[Tuple[str, frozenset]] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize an ChemicalLoss instance."""
        if len(self.atom_idxs_keys) != 4:
            raise ValueError(f"ChemicalLoss requires exactly 4 atom keys, got {len(self.atom_idxs_keys)}.")
        
        if not self.kde_file:
            raise ValueError(f"Subclass {self.__class__.__name__} must specify kde_file attribute")
        
        if not self.linkage_pdb_file:
            raise ValueError(f"Subclass {self.__class__.__name__} must specify linkage_pdb_file attribute")
        
        if not method:
            raise ValueError("Method string cannot be empty")
            
        if set(atom_idxs.keys()) != set(self.atom_idxs_keys):
            raise ValueError(f"Expected atom keys {self.atom_idxs_keys}, got {set(atom_idxs.keys())}")
        
        # Store parameters as torch tensors where possible for efficiency
        self._method = method
        self._atom_idxs = atom_idxs
        self._temp = torch.tensor(temp, device=device)
        self._weight = torch.tensor(weight, device=device)
        self._offset = torch.tensor(offset, device=device)
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load KDE model
        kde_manager = KDEManager()
        self.kde_pdf = kde_manager.get_kde(self.kde_file, self._device)
        
        # Resonance tracking
        self._resonance_key = resonance_key
        
        # Pre-compute vertex indices as tensor for efficiency
        self._vertex_indices = torch.tensor(
            [self._atom_idxs[k] for k in self.atom_idxs_keys],
            dtype=torch.long,
            device=self._device
        )
    
    @property
    def method(self) -> str:
        return self._method
    
    @property
    def resonance_key(self) -> Optional[Tuple[str, frozenset]]:
        return self._resonance_key
    
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate loss for a batch of atom positions (in Angstroms).
        Optimized for torch.jit compilation.
        """
        # Extract tetrahedral vertex positions using pre-computed indices
        vertex_positions = positions[:, self._vertex_indices, :]
        
        # Stack atom positions for vectorized distance calculation
        v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
        
        atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
        atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
        
        # Calculate pairwise distances
        dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
        
        # Evaluate log probability using KDE
        logP = self.kde_pdf.score_samples(dists)
        
        # Convert to energy using Boltzmann relation
        return -KB * self._temp * logP
    
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        warnings.warn("ChemicalLoss instances should not be called directly. Use ChemicalLossHandler instead")
        """Compute total loss from atom positions."""
        loss = self._eval_loss(positions)
        return loss * self._weight + self._offset
    
    @staticmethod
    def find_valid_pairs(
        traj: md.Trajectory,
        atom_indexes_dict: Dict,
        donor_residue_names: Union[str, List[str]],
        acceptor_residue_names: Union[str, List[str]],
        donor_atom_groups: Dict[str, List[str]],
        acceptor_atom_groups: Dict[str, List[str]],
        method_name: str,
        exclude_residue_names: Optional[List[str]] = None,
        require_terminals: bool = False,
        special_selection: Optional[callable] = None,
    ) -> List[IndexesMethodPair]:
        """
        A utility method to find valid residue pairs for a given cyclization chemistry.
        """
        # Convert single strings to lists for consistent handling
        if isinstance(donor_residue_names, str):
            donor_residue_names = [donor_residue_names]
        if isinstance(acceptor_residue_names, str):
            acceptor_residue_names = [acceptor_residue_names]
        
        # Default exclude list if none provided
        exclude_residue_names = exclude_residue_names or list(AMBER_CAPS)
        
        # Get all residues excluding specified ones
        all_residues = list(traj.topology.residues)
        valid_residues = [r for r in all_residues if r.name not in exclude_residue_names]
        
        # Check if we have enough residues for terminal selection
        if require_terminals and len(valid_residues) < 2:
            warnings.warn(f"[{method_name}] Not enough residues to define terminals.")
            return []
        
        # Handle terminal residue identification if needed
        n_term = valid_residues[0] if require_terminals else None
        c_term = valid_residues[-1] if require_terminals else None
        
        # Filter residues by type
        donor_residues = [r for r in valid_residues if r.name in donor_residue_names]
        acceptor_residues = [r for r in valid_residues if r.name in acceptor_residue_names]
        
        # Check if we found residues of the required types
        if not donor_residues:
            warnings.warn(f"[{method_name}] No {'/'.join(donor_residue_names)} residues found.")
            return []
        if not acceptor_residues:
            warnings.warn(f"[{method_name}] No {'/'.join(acceptor_residue_names)} residues found.")
            return []
        
        # Determine which residue pairs to evaluate
        residue_pairs = []
        
        # Use custom selection function if provided
        if special_selection:
            residue_pairs = special_selection(donor_residues, acceptor_residues)
        # Otherwise do a standard cross-product of donors and acceptors
        else:
            residue_pairs = [(d, a) for d in donor_residues for a in acceptor_residues if d.index != a.index]
        
        # For terminal-specific methods, override pairs
        if require_terminals:
            if "n_term" in donor_atom_groups:
                # N-terminal to something
                residue_pairs = [(n_term, a) for a in acceptor_residues if n_term.index != a.index]
            elif "c_term" in acceptor_atom_groups:
                # Something to C-terminal
                residue_pairs = [(d, c_term) for d in donor_residues if d.index != c_term.index]
            elif "n_term" in acceptor_atom_groups:
                # Something to N-terminal
                residue_pairs = [(d, n_term) for d in donor_residues if d.index != n_term.index]
            elif "c_term" in donor_atom_groups:
                # C-terminal to something
                residue_pairs = [(c_term, a) for a in acceptor_residues if c_term.index != a.index]
        
        # List to collect valid pairings
        valid_pairs = []
        
        # Process all candidate residue pairs
        for donor, acceptor in residue_pairs:
            # First, map each key to all possible atom names for both residues
            donor_atom_options = {}
            acceptor_atom_options = {}
            
            # Special handling for terminal residues if needed
            if require_terminals:
                # Substitute n_term or c_term placeholders with actual atom keys
                if "n_term" in donor_atom_groups and donor == n_term:
                    donor_atom_groups = donor_atom_groups.copy()
                    n_term_atoms = donor_atom_groups.pop("n_term")
                    for key, atoms in n_term_atoms.items():
                        donor_atom_groups[key] = atoms
                if "c_term" in acceptor_atom_groups and acceptor == c_term:
                    acceptor_atom_groups = acceptor_atom_groups.copy()
                    c_term_atoms = acceptor_atom_groups.pop("c_term")
                    for key, atoms in c_term_atoms.items():
                        acceptor_atom_groups[key] = atoms
            
            # Build a dictionary of all atom permutations
            for key, atom_names in donor_atom_groups.items():
                donor_atom_options[key] = [atom_indexes_dict.get((donor.index, name)) for name in atom_names 
                                        if (donor.index, name) in atom_indexes_dict]
                if not donor_atom_options[key]:
                    warnings.warn(f"[{method_name}] {donor.name} {donor.index} missing required atom(s) {atom_names}")
                    break
            else:  # Only executed if no break occurred
                for key, atom_names in acceptor_atom_groups.items():
                    acceptor_atom_options[key] = [atom_indexes_dict.get((acceptor.index, name)) for name in atom_names
                                                if (acceptor.index, name) in atom_indexes_dict]
                    if not acceptor_atom_options[key]:
                        warnings.warn(f"[{method_name}] {acceptor.name} {acceptor.index} missing required atom(s) {atom_names}")
                        break
                else:  # Only executed if inner loop completed without break
                    # Generate all combinations
                    resonance_keys = []
                    resonance_values = []
                    
                    # Collect all keys and their possible values for combinatorial generation
                    for key, indices in donor_atom_options.items():
                        resonance_keys.append(key)
                        resonance_values.append(indices)
                    for key, indices in acceptor_atom_options.items():
                        resonance_keys.append(key)
                        resonance_values.append(indices)
                    
                    # Generate all combinations using itertools.product
                    for values in itertools.product(*resonance_values):
                        # Create atom_dict for this permutation
                        atom_dict = {key: value for key, value in zip(resonance_keys, values)}
                        
                        # Generate resonance info string for the method name
                        resonance_info = []
                        for key, atom_names in donor_atom_groups.items():
                            if len(atom_names) > 1:
                                idx = donor_atom_options[key].index(atom_dict[key])
                                resonance_info.append(f"{atom_names[idx]}")
                        for key, atom_names in acceptor_atom_groups.items():
                            if len(atom_names) > 1:
                                idx = acceptor_atom_options[key].index(atom_dict[key])
                                resonance_info.append(f"{atom_names[idx]}")
 
                        # Format the method string
                        if resonance_info:
                            resonance_str = f" (resonant: {', '.join(resonance_info)})"
                        else:
                            resonance_str = ""
                        
                        # Handle special cases for terminals
                        if donor == n_term and acceptor == c_term:
                            method_str = f"{method_name}, N-term {donor.name} {donor.index} -> C-term {acceptor.name} {acceptor.index}{resonance_str}"
                        elif donor == n_term:
                            method_str = f"{method_name}, N-term {donor.name} {donor.index} -> {acceptor.name} {acceptor.index}{resonance_str}"
                        elif acceptor == c_term:
                            method_str = f"{method_name}, {donor.name} {donor.index} -> C-term {acceptor.name} {acceptor.index}{resonance_str}"
                        elif donor == c_term:
                            method_str = f"{method_name}, C-term {donor.name} {donor.index} -> {acceptor.name} {acceptor.index}{resonance_str}"
                        elif acceptor == n_term:
                            method_str = f"{method_name}, {donor.name} {donor.index} -> N-term {acceptor.name} {acceptor.index}{resonance_str}"
                        else:
                            method_str = f"{method_name}, {donor.name} {donor.index} -> {acceptor.name} {acceptor.index}{resonance_str}"
                        
                        # Create the IndexesMethodPair and add to results
                        valid_pairs.append(IndexesMethodPair(atom_dict, method_str, {donor.index, acceptor.index}))
        
        return valid_pairs
    
    @abstractmethod
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Identify valid atom configurations for this loss in a structure.
        Must be implemented by subclasses.
        
        Args:
            traj: MDTraj trajectory for analysis and atom identification
            atom_indexes_dict: Dictionary mapping (residue_idx, atom_name) to global atom indices
            
        Returns:
            List of IndexesMethodPair objects.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_indexes_and_methods") 
    
    @abstractmethod
    def _build_final_structure(self,
                              initial_structure: pmd.Structure) -> pmd.Structure:
        """
        Builds a final structure, assuming this cyclization has already occurred. 
        
        This method should:
        1. Create the cyclization bond(s) using the linkage information
        2. Remove/modify atoms as needed for the specific chemistry
        3. Update the topology to reflect the new connectivity
        4. Preserve coordinates where possible
        
        The returned structure will likely need geometry optimization but should have
        correct connectivity and topology.
        
        Args:
            initial_structure: ParmED Structure object representing the initial state
            
        Returns:
            ParmED Structure object with the cyclization applied
            
        Note:
            This structure will almost certainly need to be optimized, and this will not be handled here.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement _build_final_structure")

    @final
    def _validate_initial_structure(self, initial_structure: pmd.Structure) -> None:
        """
        Validates the initial ParmED structure.
        
        Args:
            initial_structure: ParmED Structure to validate
            
        Raises:
            ValueError: If the structure is invalid for cyclization
            TypeError: If the input is not a ParmED Structure
        """
        if not isinstance(initial_structure, pmd.Structure):
            raise TypeError("Initial structure must be a ParmED Structure object")
        
        if len(initial_structure.atoms) == 0:
            raise ValueError("Initial structure contains no atoms")
            
        # Check that required atoms exist
        atom_indices = [self._atom_idxs[key] for key in self.atom_idxs_keys]
        max_idx = max(atom_indices)
        if max_idx >= len(initial_structure.atoms):
            raise ValueError(f"Atom index {max_idx} exceeds structure size ({len(initial_structure.atoms)} atoms)")
        
    @staticmethod
    def _remove_hydrogens_from_atoms(initial_structure: pmd.Structure, 
                                     atom_idxs: List[int],
                                     remake: bool = True) -> pmd.Structure:
        """
        Removes hydrogens from the relevant atoms of the structure.

        Then remake the structure to update indices after atom removal.
        """
        # Collect hydrogen atoms to remove and their associated bonds
        hydrogens_to_remove = []
        bonds_to_remove = []
        
        # 1. Find all hydrogen atoms bonded to the input atoms
        for atom_idx in atom_idxs:
            for bonded_atom in initial_structure.atoms[atom_idx].bond_partners:
                if bonded_atom.element_symbol == 'H':
                    hydrogens_to_remove.append(bonded_atom)
                    
                    # Find all bonds involving this hydrogen atom
                    for bond in initial_structure.bonds:
                        if bonded_atom in (bond.atom1, bond.atom2):
                            bonds_to_remove.append(bond)
        
        # 2. Remove bonds first (to avoid reference issues)
        for bond in bonds_to_remove:
            if bond in initial_structure.bonds:
                initial_structure.bonds.remove(bond)
        
        # 3. Remove hydrogen atoms (in reverse order to maintain indices)
        for h_atom in sorted(hydrogens_to_remove, key=lambda x: x.idx, reverse=True):
            initial_structure.atoms.pop(h_atom.idx)
        
        # 4. Rebuild the structure to update indices after atom removal
        if remake:
            initial_structure.remake()

        return initial_structure
    
    @final 
    def build_final_structure(self,
                              initial_structure: pmd.Structure) -> pmd.Structure:
        """
        Builds a final structure, assuming this cyclization has already occurred.
        
        This is the public interface that adds validation and error handling around
        the abstract _build_final_structure method.
        
        Args:
            initial_structure: ParmED Structure object representing the initial state
            
        Returns:
            ParmED Structure object with the cyclization applied and validated
            
        Raises:
            ValueError: If the initial structure is invalid
            TypeError: If the input is not a ParmED Structure
        """
        self._validate_initial_structure(initial_structure)
        
        try:
            final_structure = self._build_final_structure(initial_structure)
            
            # Validate the output
            if not isinstance(final_structure, pmd.Structure):
                raise TypeError("_build_final_structure must return a ParmED Structure object")
                
            return final_structure
            
        except Exception as e:
            raise RuntimeError(f"Failed to build final structure for {self.__class__.__name__}: {str(e)}") from e
    
    @staticmethod
    def structure_from_trajectory(traj: md.Trajectory, frame: int = 0) -> pmd.Structure:
        """
        Utility method to convert an MDTraj trajectory to a ParmED Structure.
        
        Args:
            traj: MDTraj trajectory
            frame: Frame index to extract (default: 0)
            
        Returns:
            ParmED Structure object
        """
        if frame >= traj.n_frames:
            raise ValueError(f"Frame {frame} exceeds trajectory length ({traj.n_frames} frames)")
            
        # Use temporary directory for safe file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdb = os.path.join(temp_dir, "temp_structure.pdb")
            traj[frame].save_pdb(temp_pdb)
            structure = pmd.load_file(temp_pdb)
            return structure
    
    @staticmethod  
    def trajectory_from_structure(structure: pmd.Structure) -> md.Trajectory:
        """
        Utility method to convert a ParmED Structure to an MDTraj trajectory.
        
        Args:
            structure: ParmED Structure object
            
        Returns:
            MDTraj trajectory (single frame)
        """
        # Use temporary directory for safe file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdb = os.path.join(temp_dir, "temp_trajectory.pdb")
            structure.save(temp_pdb)
            traj = md.load(temp_pdb)
            return traj
        
    @staticmethod
    def structure_to_pdb(structure: pmd.Structure, filename: str) -> None:
        """
        Utility method to convert a ParmED Structure to a PDB file.
        """
        structure.save(filename, format='pdb')
    
    # these are sloppy as we reuse alot of code. Let's clean this up later.
    @staticmethod
    def _remove_amber_caps(initial_structure: pmd.Structure, 
                           remake: bool = True,
                           verbose: bool = False) -> pmd.Structure:
        """
        Removes Amber cap atoms if they are present (ACE, NME, NHE)
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get first and last residues
        first_residue = structure.residues[0]
        last_residue = structure.residues[-1]
        
        # Track atoms to remove
        atoms_to_remove = []
        
        # Check if first residue is an amber cap
        if first_residue.name in AMBER_CAPS:
            atoms_to_remove.extend(list(first_residue.atoms))
        
        # Check if last residue is an amber cap (and not the same as first)
        if last_residue != first_residue and last_residue.name in AMBER_CAPS:
            atoms_to_remove.extend(list(last_residue.atoms))
        
        # Remove atoms in reverse order to maintain indices
        for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
            structure.atoms.pop(atom.idx)
        
        # Rebuild the structure to update indices after atom removal
        if remake:
            structure.remake()
        
        return structure
    
    @staticmethod
    def _remove_amber_head(initial_structure: pmd.Structure, 
                           remake: bool = True, 
                           verbose: bool = False,
                           ) -> pmd.Structure:
        """
        Removes any Amber caps found in the first residue
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get the first residue
        first_residue = structure.residues[0]
        
        # Check if first residue is an amber cap
        if first_residue.name in AMBER_CAPS:
            # Get all atoms in the first residue
            atoms_to_remove = list(first_residue.atoms)
            
            # Remove atoms in reverse order to maintain indices
            for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
                structure.atoms.pop(atom.idx)
            
            # Rebuild the structure to update indices after atom removal
            if remake:
                structure.remake()
        
        return structure
    
    @staticmethod
    def _remove_amber_tail(initial_structure: pmd.Structure, 
                           remake: bool = True, 
                           verbose: bool = False,
                           ) -> pmd.Structure:
        """
        Removes any Amber caps found in the last residue
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get the last residue
        last_residue = structure.residues[-1]
        
        # Check if last residue is an amber cap
        if last_residue.name in AMBER_CAPS:
            # Get all atoms in the last residue
            atoms_to_remove = list(last_residue.atoms)
            
            # Remove atoms in reverse order to maintain indices
            for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
                structure.atoms.pop(atom.idx)
            
            # Rebuild the structure to update indices after atom removal
            if remake:
                structure.remake()
        
        return structure