from typing import Dict, List, Optional, Set, Type, Union, Tuple
import torch
import mdtraj as md
from pathlib import Path

from ..core.loss_handler import LossHandler
from ..utils.constants import KB, UNITS_FACTORS_DICT
from ..utils.default_strategies import DEFAULT_STRATEGIES
from ..utils.utils import soft_min
from .eff_chemical_loss import EffChemicalLoss

class EffChemicalLossHandler(LossHandler):
    """
    Efficient version of ChemicalLossHandler optimized for torch.jit compilation and batch processing.
    This class is designed to be immutable after initialization.
    """
    
    # Default set of cyclization chemistries to consider
    default_strategies: Set[Type[EffChemicalLoss]] = DEFAULT_STRATEGIES
    
    # Common atom names involved in cyclization
    bonding_atoms = {
        "N", "CA", "C", "O", "OXT",
        "OD1", "OD2", "OE1", "OE2", "CG", "CD",
        "SG", "CB", "NE", "CZ", "NH1", "NH2",
        "NZ", "CE", "OH", "CZ"
    }
    
    # Unit conversion factors
    units_factors_dict = UNITS_FACTORS_DICT
    
    @classmethod
    def from_pdb(cls, 
                pdb_path: Union[str, Path],
                units: Optional[str] = None,
                units_factor: Optional[float] = None, 
                temp: float = 1.0,
                alpha: float = -3.0,
                mask: Optional[Set[int]] = None,
                device: Optional[torch.device] = None,
                **kwargs) -> "EffChemicalLossHandler":
        """Create an EffChemicalLossHandler from a PDB file with simplified parameters."""
        if units is not None and units_factor is not None:
            raise ValueError(
                "Provide either 'units' or 'units_factor', but not both."
            )
        
        if units is None and units_factor is None:
            raise ValueError(
                "Either 'units' or 'units_factor' must be provided."
            )
        
        if units is not None:
            try:
                units_factor = cls.units_factors_dict[units]
            except KeyError:
                valid_units = ', '.join(f"'{u}'" for u in cls.units_factors_dict.keys())
                raise ValueError(f"Unknown unit: '{units}'. Valid units are: {valid_units}")
        
        return cls(
            pdb_path=pdb_path,
            units_factor=units_factor,
            temp=temp,
            alpha=alpha,
            mask=mask,
            device=device,
            **kwargs
        )

    def __init__(self,
                 pdb_path: Union[str, Path],
                 units_factor: float,
                 strategies: Optional[Set[Type[EffChemicalLoss]]] = None,
                 weights: Optional[Dict[Type[EffChemicalLoss], float]] = None,
                 offsets: Optional[Dict[Type[EffChemicalLoss], float]] = None,
                 temp: float = 1.0,
                 alpha: float = -3.0,
                 mask: Optional[Set[int]] = None,
                 device: Optional[torch.device] = None,
                 ):
        """Initialize an EffChemicalLossHandler with detailed control over parameters."""
        super().__init__(units_factor)
        self._pdb_path = Path(pdb_path)
        
        if not self._pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {self._pdb_path}")
        
        # Load all strategy classes if not provided
        if strategies is None:
            strategies = self.default_strategies
            
        self._strategies = strategies
        self._weights = weights or {s: 1.0 for s in self._strategies}
        self._offsets = offsets or {s: 0.0 for s in self._strategies}
        
        # Validate temperature
        assert temp > 0, f"Temperature (temp) must be positive, but got {temp}."
        self._temp = temp
        self._alpha = alpha

        # Set device
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trajectory and prepare topology
        self._traj = md.load(str(self._pdb_path))
        self._topology = self._traj.topology
        
        self._mask: Set[int] = mask or set()
        self._validate_mask()

        # Initialize losses and build resonance groups
        self._initialize_losses()
        
        # Pre-compute tensors for efficiency
        self._precompute_tensors()
    
    def _precompute_tensors(self) -> None:
        """Pre-compute tensors for efficient batch processing."""
        # Convert resonance groups to tensors for efficient processing
        self._resonance_groups_tensor = []
        self._resonance_groups_indices = []
        
        for resonance_key, losses_in_group in self._resonance_groups.items():
            # Store indices for each loss in the group
            group_indices = []
            for loss in losses_in_group:
                group_indices.append(loss._vertex_indices)
            self._resonance_groups_indices.append(torch.stack(group_indices))
        
        # Convert to tensors for efficient processing
        self._resonance_groups_indices = torch.stack(self._resonance_groups_indices)
    
    @torch.jit.script_method
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the loss for a batch of atom positions.
        Optimized for torch.jit compilation and batch processing.
        """
        if not self._resonance_groups:
            return torch.zeros(positions.shape[0], device=self._device)
        
        batch_size = positions.shape[0]
        group_losses = []
        
        # Process each resonance group
        for group_idx, (resonance_key, losses_in_group) in enumerate(self._resonance_groups.items()):
            # Get pre-computed indices for this group
            group_indices = self._resonance_groups_indices[group_idx]
            
            # Compute losses for all instances in this resonance group
            group_loss_values = []
            
            for loss_idx, loss in enumerate(losses_in_group):
                # Extract vertex positions using pre-computed indices
                vertex_indices = group_indices[loss_idx]
                vertex_positions = positions[:, vertex_indices, :]
                
                # Stack atom positions for vectorized distance calculation
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                # Evaluate KDE and convert to energy
                logP = loss.kde_pdf.score_samples(dists)
                energy = -KB * self._temp * logP
                
                # Apply weight and offset
                scaled_energy = energy * loss._weight + loss._offset
                group_loss_values.append(scaled_energy)
            
            # Take minimum across all resonant structures in this group
            if len(group_loss_values) == 1:
                group_min = group_loss_values[0]
            else:
                group_stack = torch.stack(group_loss_values, dim=1)  # [batch_size, n_resonant]
                group_min = torch.min(group_stack, dim=1)[0]  # [batch_size]
            
            group_losses.append(group_min)
        
        # Apply soft minimum across all unique cyclization groups
        if len(group_losses) == 1:
            final_loss = group_losses[0]
        else:
            all_group_losses = torch.stack(group_losses, dim=1)  # [batch_size, n_groups]
            final_loss = soft_min(all_group_losses, alpha=self._alpha)
        
        return final_loss
    
    def _validate_mask(self) -> None:
        """Validate the mask indices based on the topology."""
        if self._mask:
            valid_indices = {res.index for res in self._topology.residues}
            invalid_indices = self._mask - valid_indices
            if invalid_indices:
                raise ValueError(
                    f"Invalid residue indices in mask: {invalid_indices}. "
                    f"Valid indices range from 0 to {len(valid_indices)-1}."
                )
    
    def _initialize_losses(self) -> None:
        """Initialize all loss functions and organize them into resonance groups."""
        if not self.strategies:
            raise ValueError(
                "No cyclization strategies provided and default_strategies is empty."
            )

        # Get the loaded trajectory
        traj = self.traj
        
        # Pre-compute atom indices for faster lookup
        atom_idx_dict = self.precompute_atom_indices(
            list(self.topology.residues),
            self.bonding_atoms
        )
        
        # Dictionary to group losses by resonance key
        resonance_groups: Dict[Tuple[str, frozenset], List[EffChemicalLoss]] = {}
        
        # Create all loss instances
        for strat in self.strategies:
            for idxs_method_pair in strat.get_indexes_and_methods(traj, atom_idx_dict):
                # Skip if either residue in the pair is masked
                if self._mask and (idxs_method_pair.pair & self._mask):
                    continue
                
                # Extract the base method (without resonance info) for grouping
                method_base = idxs_method_pair.method.split(" (")[0]
                resonance_key = (method_base, frozenset(idxs_method_pair.pair))
                
                # Create loss instance
                loss = strat(
                    method=idxs_method_pair.method,
                    atom_idxs=idxs_method_pair.indexes,
                    temp=self._temp,
                    weight=self._weights[strat],
                    offset=self._offsets[strat],
                    resonance_key=resonance_key,
                    device=self._device,
                )
                
                # Group by resonance key
                if resonance_key not in resonance_groups:
                    resonance_groups[resonance_key] = []
                resonance_groups[resonance_key].append(loss)
        
        # Store the grouped structure for efficient evaluation
        self._resonance_groups = resonance_groups
        
        # Also maintain a flat list for compatibility
        self._losses = []
        for group in resonance_groups.values():
            self._losses.extend(group)
    
    @staticmethod
    def precompute_atom_indices(residues, atom_names) -> Dict[Tuple[int, str], int]:
        """Precompute a dictionary mapping (residue_index, atom_name) to atom indices."""
        indices = {}
        for residue in residues:
            for name in atom_names:
                atom = next((a for a in residue.atoms if a.name == name), None)
                if atom:
                    indices[(residue.index, name)] = atom.index
        return indices
    
    @property
    def pdb_path(self) -> Path:
        return self._pdb_path
    
    @property
    def strategies(self) -> Set[Type[EffChemicalLoss]]:
        return self._strategies
    
    @property
    def weights(self) -> Dict[Type[EffChemicalLoss], float]:
        return self._weights
    
    @property
    def offsets(self) -> Dict[Type[EffChemicalLoss], float]:
        return self._offsets
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @property
    def temp(self) -> float:
        return self._temp
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def losses(self) -> List[EffChemicalLoss]:
        return self._losses
    
    @property
    def traj(self) -> md.Trajectory:
        return self._traj
    
    @property
    def topology(self) -> md.Topology:
        return self._topology 