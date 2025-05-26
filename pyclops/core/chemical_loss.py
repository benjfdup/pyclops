from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, ClassVar, Callable, Union, final
import warnings
import logging
import os.path

import torch
import mdtraj as md

from ..torchkde import KernelDensity  # this is, itself, a torch.nn.module
#torch.serialization.add_safe_globals([KernelDensity])

from ..utils.indexing import IndexesMethodPair
from ..utils.constants import KB, AMBER_CAPS


# Configure logging
logger = logging.getLogger("pyclops.chemical_loss")

class KDEManager:
    """Singleton manager for KDE models to prevent redundant loading"""
    
    _instance = None
    _kde_cache: Dict[str, KernelDensity] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KDEManager, cls).__new__(cls)
        return cls._instance
    
    def get_kde(self, kde_file: str, device: torch.device) -> KernelDensity:
        """
        Get or load a KDE model from cache using the provided file path.
        
        Args:
            kde_file: Full path to the KDE file (with .pt extension)
            device: PyTorch device to load the model on
            
        Returns:
            Loaded KernelDensity model
            
        Raises:
            FileNotFoundError: If the KDE model file does not exist
        """
        # Return cached model if available
        if kde_file in self._kde_cache:
            return self._kde_cache[kde_file]
        
        # Check if file exists
        if not os.path.exists(kde_file):
            error_msg = f"KDE model file does not exist: '{kde_file}'"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load the model
        logger.debug(f"Loading KDE model from file: {kde_file}")
        self._kde_cache[kde_file] = torch.load(kde_file, map_location=device, weights_only=False) # make more safe in future.
        return self._kde_cache[kde_file]
    
    def clear_cache(self) -> None:
        """Clear the KDE model cache to free memory"""
        self._kde_cache.clear()
        logger.debug("KDE model cache cleared")


class ChemicalLoss(ABC):
    """
    Abstract base class representing a chemical loss function derived from a tetrahedral
    Boltzmann distribution.

    This class provides a common interface and logic for evaluating chemical constraints 
    based on statistical energy potentials estimated via KDE. The loss guides the geometry 
    of a specific subset of atoms toward physically meaningful configurations.

    However, this class should rarely, and perhaps never, be instantiated directly.

    Subclasses must define:
    - `atom_idxs_keys`: Four atom keys corresponding to the tetrahedron vertices
    - `kde_file`: Full path to the KDE model file (.pt file) - MANDATORY
    - `get_indexes_and_methods()`: Method to identify valid configurations in a structure
    
    Attributes:
        atom_idxs_keys (ClassVar[List[str]]): The four atom keys for the tetrahedral geometry
        kde_file (ClassVar[str]): Full path to the KDE model file (with .pt extension)
    """
    # Class variables to be overridden by subclasses
    atom_idxs_keys: ClassVar[List[str]] = []
    kde_file: ClassVar[str] = ''  # Mandatory attribute
    
    def __init__(
        self,
        method: str, 
        atom_idxs: Dict[str, int],
        temp: float = 300.0,
        weight: float = 1.0,
        offset: float = 0.0,
        resonance_key: Optional[Tuple[str, frozenset]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a ChemicalLoss instance.
        
        Args:
            method: Descriptive string for this cyclization chemistry
            atom_idxs: Mapping from atom keys to indices in the structure
            temp: Temperature (K) for Boltzmann scaling
            weight: Weight multiplier for the loss
            offset: Constant bias added to the loss
            resonance_key: For resonance tracking by subclasses
            device: PyTorch device for computation
        
        Raises:
            ValueError: If atom keys or KDE file are invalid
        """
        # Validate class configuration
        if len(self.atom_idxs_keys) != 4:
            raise ValueError(
                f"ChemicalLosses require exactly 4 atom keys, got {len(self.atom_idxs_keys)}."
            )
        
        # Check for kde_file (mandatory)
        if not self.kde_file:
            raise ValueError(
                f"Subclass {self.__class__.__name__} must specify kde_file attribute pointing to a .pt file"
            )
        
        # Validate instance parameters
        if not method:
            raise ValueError("Method string cannot be empty")
            
        if set(atom_idxs.keys()) != set(self.atom_idxs_keys):
            raise ValueError(
                f"Expected atom keys {self.atom_idxs_keys}, got {set(atom_idxs.keys())}"
            )
        
        # Store parameters
        self._method = method
        self._atom_idxs = atom_idxs
        self._temp = temp
        self._weight = weight
        self._offset = offset
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load KDE model using the singleton manager
        kde_manager = KDEManager()
        self.kde_pdf = kde_manager.get_kde(self.kde_file, self._device)
        
        # Resonance tracking (used by subclasses)
        self._resonance_key: Optional[Tuple[str, frozenset]] = resonance_key
    
    # The rest of the ChemicalLoss class remains unchanged
    @property
    def temp(self) -> float:
        """Temperature (K) used in the Boltzmann loss calculation"""
        return self._temp
    
    @property
    def atom_idxs(self) -> Dict[str, int]:
        """Atom indices used for loss calculation"""
        return self._atom_idxs
    
    @property
    def weight(self) -> float:
        """Weight multiplier for the loss"""
        return self._weight
    
    @property
    def offset(self) -> float:
        """Constant bias added to the loss"""
        return self._offset
    
    @property
    def method(self) -> str:
        """Descriptive string of the cyclization chemistry"""
        return self._method
    
    @property
    def device(self) -> torch.device:
        """Computation device"""
        return self._device
    
    @property
    def resonance_key(self) -> Optional[Tuple[str, frozenset]]:
        """Key for identifying resonant structures"""
        return self._resonance_key
    @resonance_key.setter
    def resonance_key(self, res_key):
        self._resonance_key = res_key
    
    @final
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate loss for a batch of atom positions (in Angstroms).
        
        Args:
            positions: Tensor of shape [batch_size, n_atoms, 3]
            
        Returns:
            Tensor of shape [batch_size] containing loss values
        """
        # Extract tetrahedral vertex positions
        v0 = positions[:, self.atom_idxs[self.atom_idxs_keys[0]], :].squeeze()
        v1 = positions[:, self.atom_idxs[self.atom_idxs_keys[1]], :].squeeze()
        v2 = positions[:, self.atom_idxs[self.atom_idxs_keys[2]], :].squeeze()
        v3 = positions[:, self.atom_idxs[self.atom_idxs_keys[3]], :].squeeze()
        
        # Stack atom positions for vectorized distance calculation
        atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
        atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
        
        # Calculate pairwise distances (6 distances for tetrahedron)
        dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
        
        # Evaluate log probability using KDE
        logP = self.kde_pdf.score_samples(dists)
        
        # Convert to energy using Boltzmann relation
        return -KB * self.temp * logP
    
    @final
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss from atom positions.
        
        Args:
            positions: Tensor of shape [batch_size, n_atoms, 3] in Angstroms
            
        Returns:
            Tensor of shape [batch_size] with weighted, offset loss values
        """
        loss = self._eval_loss(positions)
        return loss * self.weight + self.offset

    def get_vertexes_atomic_idxs(self) -> torch.Tensor:
        """
        Get atom indices for the tetrahedral vertices.
        
        Returns:
            Tensor of shape [4] with atom indices
        """
        return torch.tensor(
            [self._atom_idxs[k] for k in self.atom_idxs_keys],
            dtype=torch.long,
            device=self.device
        )
    
    def get_vertex_atomic_idxs(self) -> torch.Tensor: # alias, for my convenience (maybe will clean up later)
        """
        Get atom indices for the tetrahedral vertices.
        
        Returns:
            Tensor of shape [4] with atom indices
        """
        return self.get_vertexes_atomic_idxs()
    
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
        special_selection: Optional[Callable] = None,
    ) -> List[IndexesMethodPair]:
        """
        A utility method to find valid residue pairs for a given cyclization chemistry.
        
        This method handles the common patterns in get_indexes_and_methods implementations:
        - Finding residues by type
        - Handling resonance structures
        - Creating indexed method strings
        - Warning on missing atoms
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing the structure.
            
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_idx, atom_name) to atom index.
            
        donor_residue_names : Union[str, List[str]]
            Name(s) of the donor residue type(s) (e.g., "CYS" or ["ASP", "GLU"]).
            
        acceptor_residue_names : Union[str, List[str]]
            Name(s) of the acceptor residue type(s).
            
        donor_atom_groups : Dict[str, List[str]]
            Dictionary mapping atom key names to possible atom names in the donor residue.
            Each list of atom names represents resonant alternatives for the same chemical role.
            Example: {'S1': ['SG'], 'C1': ['CB']}
            
        acceptor_atom_groups : Dict[str, List[str]]
            Dictionary mapping atom key names to possible atom names in the acceptor residue.
            Example: {'O1': ['OD1', 'OD2'], 'C1': ['CG']}
            
        method_name : str
            Name of the cyclization method for reporting.
            
        exclude_residue_names : List[str], optional
            Residue names to exclude (e.g., capping residues like "ACE", "NME").
            
        require_terminals : bool, optional
            If True, will attempt to find and use terminal residues.
            
        special_selection : Callable, optional
            Optional function that takes (donor_residues, acceptor_residues) and returns
            a list of (donor, acceptor) tuples to consider. This allows custom filtering
            beyond simple residue type matching.
            
        Returns
        -------
        List[IndexesMethodPair]
            List of valid IndexesMethodPair objects for the specified chemistry.
            
        Examples
        --------
        >>> # Finding disulfide bonds
        >>> pairs = ChemicalLoss.find_valid_pairs(
        ...     traj, atom_indexes_dict,
        ...     donor_residue_names="CYS", 
        ...     acceptor_residue_names="CYS",
        ...     donor_atom_groups={'S1': ['SG'], 'C1': ['CB']},
        ...     acceptor_atom_groups={'S2': ['SG'], 'C2': ['CB']},
        ...     method_name="Disulfide",
        ...     # Custom selection to avoid self-pairing and duplicates
        ...     special_selection=lambda d, a: [(d[i], a[j]) for i in range(len(d)) for j in range(i+1, len(a))]
        ... )
        
        >>> # Finding amide bonds with resonant oxygens
        >>> pairs = ChemicalLoss.find_valid_pairs(
        ...     traj, atom_indexes_dict,
        ...     donor_residue_names="LYS", 
        ...     acceptor_residue_names="GLU",
        ...     donor_atom_groups={'N1': ['NZ'], 'C2': ['CE']},
        ...     acceptor_atom_groups={'C1': ['CD'], 'O1': ['OE1', 'OE2']},
        ...     method_name="AmideLysGlu"
        ... )
        """
        # Convert single strings to lists for consistent handling
        if isinstance(donor_residue_names, str):
            donor_residue_names = [donor_residue_names]
        if isinstance(acceptor_residue_names, str):
            acceptor_residue_names = [acceptor_residue_names]
        
        # Default exclude list if none provided
        exclude_residue_names = exclude_residue_names or list(AMBER_CAPS) # residues to exclude
        
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
                    import itertools
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
    
    @classmethod
    @abstractmethod #TODO: Think about adding some standardization here...
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Identify valid atom configurations for this loss in a structure.

        Generally, these methods should call the `find_valid_pairs` static method of `ChemicalLoss`, though
        this is not enforced nor 
        
        Args:
            traj: MDTraj trajectory with molecule structure
            atom_indexes_dict: Mapping from (residue_index, atom_name) to atom index
            
        Returns:
            List of IndexesMethodPair objects for valid configurations.
        
        Example:
            >>> traj = md.load('peptide.pdb')
            >>> atom_dict = {(0, 'N'): 0, (0, 'CA'): 1, ...}
            >>> pairs = MyChemicalLoss.get_indexes_and_methods(traj, atom_dict)
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_indexes_and_methods")