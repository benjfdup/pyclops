from typing import Dict, List, Optional, Tuple, ClassVar, Union, final
import torch
import mdtraj as md

from ..torchkde import KernelDensity
from ..utils.indexing import IndexesMethodPair
from ..utils.constants import KB

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

class EffChemicalLoss:
    """
    Efficient version of ChemicalLoss optimized for torch.jit compilation.
    This class is designed to be immutable after initialization.
    """
    
    # Class variables to be overridden by subclasses
    atom_idxs_keys: ClassVar[List[str]] = []
    kde_file: ClassVar[str] = ''
    
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
        """Initialize an EffChemicalLoss instance."""
        if len(self.atom_idxs_keys) != 4:
            raise ValueError(f"EffChemicalLoss requires exactly 4 atom keys, got {len(self.atom_idxs_keys)}.")
        
        if not self.kde_file:
            raise ValueError(f"Subclass {self.__class__.__name__} must specify kde_file attribute")
        
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
    
    @torch.jit.script_method
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
    
    @torch.jit.script_method
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
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
        This is kept identical to the original for compatibility.
        """
        # Implementation identical to ChemicalLoss.find_valid_pairs
        # This is kept as a static method for compatibility with existing code
        from ..core.chemical_loss import ChemicalLoss
        return ChemicalLoss.find_valid_pairs(
            traj, atom_indexes_dict,
            donor_residue_names, acceptor_residue_names,
            donor_atom_groups, acceptor_atom_groups,
            method_name, exclude_residue_names,
            require_terminals, special_selection
        )
    
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Identify valid atom configurations for this loss in a structure.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_indexes_and_methods") 