from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, ClassVar, Union
import torch
import mdtraj as md
import warnings
import itertools

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
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_indexes_and_methods") 
    
    def build_final_structure(self, initial_traj: md.Trajectory, positions: torch.Tensor) -> tuple[md.Trajectory, torch.Tensor]:
        """
        Builds a final structure, assuming this cyclization has already occured. 
        This does so by replacing the relevant atoms in the initial structure with the atoms they correspond to in the 
        linkage PDB file, and then adding the remaining atoms.
        
        This final structure will almost certainly need to be optimized
        args:
            positions: torch.Tensor, shape [n_atoms, 3]
            initial_traj: md.Trajectory, shape [n_atoms, 3]
        returns:
            final_traj: md.Trajectory, shape [n_atoms, 3]
            final_positions: torch.Tensor, shape [n_atoms, 3]
        """        