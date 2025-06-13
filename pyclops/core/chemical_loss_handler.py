from typing import Dict, List, Optional, Set, Type, Union, Tuple
import torch
import mdtraj as md
from pathlib import Path

from ..core.loss_handler import LossHandler
from ..utils.constants import KB, UNITS_FACTORS_DICT
from ..utils.default_strategies import DEFAULT_STRATEGIES
from ..utils.utils import soft_min
from .chemical_loss import ChemicalLoss


class ChemicalLossHandler(LossHandler):
    """
    ChemicalLossHandler optimized for torch.jit compilation and batch processing.
    This class is designed to be immutable after initialization.
    """
    
    # Default set of cyclization chemistries to consider
    default_strategies: Set[Type[ChemicalLoss]] = DEFAULT_STRATEGIES
    
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
                **kwargs) -> "ChemicalLossHandler":
        """Create an ChemicalLossHandler from a PDB file with simplified parameters."""
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

    def _debug_print(self, message: str) -> None:
        """Helper method to print debug messages if debug is enabled."""
        if self._debug:
            print(f"[DEBUG] {message}")

    def __init__(self,
                 pdb_path: Union[str, Path],
                 units_factor: float,
                 strategies: Optional[Set[Type[ChemicalLoss]]] = None,
                 weights: Optional[Dict[Type[ChemicalLoss], float]] = None,
                 offsets: Optional[Dict[Type[ChemicalLoss], float]] = None,
                 temp: float = 1.0,
                 alpha: float = -3.0,
                 mask: Optional[Set[int]] = None,
                 device: Optional[torch.device] = None,
                 debug: bool = False,
                 ):
        """Initialize an ChemicalLossHandler with detailed control over parameters."""
        self._debug = debug
        self._debug_print("ChemicalLossHandler.__init__: Starting initialization")
        self._debug_print(f"PDB path: {pdb_path}")
        self._debug_print(f"Units factor: {units_factor}")
        self._debug_print(f"Temperature: {temp}K, Alpha: {alpha}")
        
        super().__init__(units_factor)
        self._pdb_path = Path(pdb_path)
        
        if not self._pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {self._pdb_path}")
        
        # Load all strategy classes if not provided
        if strategies is None:
            strategies = self.default_strategies
            self._debug_print(f"Using default strategies: {[s.__name__ for s in strategies]}")
        else:
            self._debug_print(f"Using provided strategies: {[s.__name__ for s in strategies]}")
            
        self._strategies = strategies
        self._weights = weights or {s: 1.0 for s in self._strategies}
        self._offsets = offsets or {s: 0.0 for s in self._strategies}
        
        self._debug_print(f"Strategy weights: {[(s.__name__, w) for s, w in self._weights.items()]}")
        self._debug_print(f"Strategy offsets: {[(s.__name__, o) for s, o in self._offsets.items()]}")
        
        # Validate temperature
        assert temp > 0, f"Temperature (temp) must be positive, but got {temp}."
        self._temp = temp
        self._alpha = alpha

        # Set device
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._debug_print(f"Using device: {self._device}")
        
        # Load trajectory and prepare topology
        self._debug_print(f"Loading trajectory from {self._pdb_path}")
        self._traj = md.load(str(self._pdb_path))
        self._topology = self._traj.topology
        self._debug_print(f"Loaded trajectory with {self._traj.n_atoms} atoms, {self._traj.n_residues} residues")
        
        self._mask: Set[int] = mask or set()
        if self._mask:
            self._debug_print(f"Applying mask to residues: {sorted(self._mask)}")
        self._validate_mask()

        # Initialize losses and build resonance groups
        self._debug_print("Initializing losses...")
        self._initialize_losses()
        
        # Pre-compute tensors for efficiency
        self._debug_print("Pre-computing tensors...")
        self._precompute_tensors()
        
        # Compile the evaluation method
        self._debug_print("Compiling JIT evaluation method...")
        self._compiled_eval_loss = torch.jit.script(self._eval_loss_impl)
        
        self._debug_print("ChemicalLossHandler initialization complete!")
        self._debug_print(f"Total losses created: {len(self._losses)}")
        self._debug_print(f"Total resonance groups: {len(self._resonance_groups)}")
    
    def _precompute_tensors(self) -> None:
        """Pre-compute tensors for efficient batch processing."""
        self._debug_print("Starting tensor precomputation")
        
        # Convert resonance groups to tensors for efficient processing
        self._resonance_groups_tensor = []
        self._resonance_groups_indices = []
        self._resonance_groups_weights = []
        self._resonance_groups_offsets = []
        
        self._debug_print(f"Processing {len(self._resonance_groups)} resonance groups...")
        
        for group_idx, (resonance_key, losses_in_group) in enumerate(self._resonance_groups.items()):
            method_base, residue_pair = resonance_key
            self._debug_print(f"Group {group_idx}: {method_base} | residues={sorted(residue_pair)} | variants={len(losses_in_group)}")
            
            # Store indices, weights, and offsets for each loss in the group
            group_indices = []
            group_weights = []
            group_offsets = []
            
            for loss_idx, loss in enumerate(losses_in_group):
                self._debug_print(f"   Variant {loss_idx}: weight={loss._weight}, offset={loss._offset}")
                self._debug_print(f"   Vertex indices: {loss._vertex_indices}")
                
                group_indices.append(loss._vertex_indices)
                group_weights.append(loss._weight)
                group_offsets.append(loss._offset)
            
            indices_tensor = torch.stack(group_indices)
            weights_tensor = torch.tensor(group_weights, device=self._device)
            offsets_tensor = torch.tensor(group_offsets, device=self._device)
            
            self._debug_print(f"   Created tensors - indices: {indices_tensor.shape}, weights: {weights_tensor.shape}, offsets: {offsets_tensor.shape}")
            
            self._resonance_groups_indices.append(indices_tensor)
            self._resonance_groups_weights.append(weights_tensor)
            self._resonance_groups_offsets.append(offsets_tensor)
        
        # Convert to tensors for efficient processing
        if self._resonance_groups_indices:
            # Find the maximum number of variants across all groups for padding
            max_variants = max(tensor.shape[0] for tensor in self._resonance_groups_indices)
            self._debug_print(f"Maximum variants across groups: {max_variants}")
            
            # Pad all tensors to the same size
            padded_indices = []
            padded_weights = []
            padded_offsets = []
            
            for i, (indices_tensor, weights_tensor, offsets_tensor) in enumerate(zip(
                self._resonance_groups_indices, self._resonance_groups_weights, self._resonance_groups_offsets
            )):
                current_variants = indices_tensor.shape[0]
                if current_variants < max_variants:
                    # Pad with zeros (dummy values that won't be used)
                    padding_needed = max_variants - current_variants
                    
                    # Pad indices with zeros
                    indices_padding = torch.zeros((padding_needed, 4), dtype=torch.long, device=self._device)
                    padded_indices_tensor = torch.cat([indices_tensor, indices_padding], dim=0)
                    
                    # Pad weights with zeros
                    weights_padding = torch.zeros(padding_needed, device=self._device)
                    padded_weights_tensor = torch.cat([weights_tensor, weights_padding], dim=0)
                    
                    # Pad offsets with zeros
                    offsets_padding = torch.zeros(padding_needed, device=self._device)
                    padded_offsets_tensor = torch.cat([offsets_tensor, offsets_padding], dim=0)
                    
                    self._debug_print(f"   Group {i}: padded from {current_variants} to {max_variants} variants")
                else:
                    padded_indices_tensor = indices_tensor
                    padded_weights_tensor = weights_tensor
                    padded_offsets_tensor = offsets_tensor
                    self._debug_print(f"   Group {i}: no padding needed ({current_variants} variants)")
                
                padded_indices.append(padded_indices_tensor)
                padded_weights.append(padded_weights_tensor)
                padded_offsets.append(padded_offsets_tensor)
            
            # Now we can safely stack tensors of equal size
            self._resonance_groups_indices = torch.stack(padded_indices)
            self._resonance_groups_weights = torch.stack(padded_weights)
            self._resonance_groups_offsets = torch.stack(padded_offsets)
            
            # Store the actual number of variants for each group to avoid processing padding
            self._resonance_groups_sizes = torch.tensor([
                len(list(self._resonance_groups.values())[i]) for i in range(len(self._resonance_groups))
            ], device=self._device)
            
            self._debug_print(f"Final tensor shapes:")
            self._debug_print(f"   Indices: {self._resonance_groups_indices.shape}")
            self._debug_print(f"   Weights: {self._resonance_groups_weights.shape}")
            self._debug_print(f"   Offsets: {self._resonance_groups_offsets.shape}")
            self._debug_print(f"   Sizes: {self._resonance_groups_sizes}")
        else:
            # Handle empty case
            self._resonance_groups_indices = torch.empty((0, 0, 4), dtype=torch.long, device=self._device)
            self._resonance_groups_weights = torch.empty((0, 0), dtype=torch.float, device=self._device)
            self._resonance_groups_offsets = torch.empty((0, 0), dtype=torch.float, device=self._device)
            self._resonance_groups_sizes = torch.empty((0,), dtype=torch.long, device=self._device)
            self._debug_print(f"No resonance groups found - created empty tensors")
        
        self._debug_print("Completed tensor precomputation")
    
    @staticmethod
    @torch.jit.script
    def _eval_loss_impl(
        positions: torch.Tensor,
        resonance_groups_indices: torch.Tensor,
        resonance_groups_weights: torch.Tensor,
        resonance_groups_offsets: torch.Tensor,
        resonance_groups_sizes: torch.Tensor,
        temp: float,
        alpha: float,
        device: torch.device,
        kb: float,
        log_probs: torch.Tensor  # Pre-computed log probabilities
    ) -> torch.Tensor:
        """
        Implementation of loss evaluation that will be compiled.
        """
        if resonance_groups_indices.shape[0] == 0:
            return torch.zeros(positions.shape[0], device=device)
        
        #batch_size = positions.shape[0]
        group_losses = []
        
        # Process each resonance group
        for group_idx in range(resonance_groups_indices.shape[0]):
            # Get the actual number of variants for this group
            actual_variants = int(resonance_groups_sizes[group_idx])
            
            # Get pre-computed indices, weights, and offsets for this group
            #group_indices = resonance_groups_indices[group_idx]
            group_weights = resonance_groups_weights[group_idx]
            group_offsets = resonance_groups_offsets[group_idx]
            
            # Compute losses for all instances in this resonance group (only up to actual_variants)
            group_loss_values = []
            
            for loss_idx in range(actual_variants):
                # Use pre-computed log probabilities
                logP = log_probs[group_idx, loss_idx]
                energy = -kb * temp * logP
                
                # Apply weight and offset
                weight = group_weights[loss_idx]
                offset = group_offsets[loss_idx]
                scaled_energy = weight * energy + offset
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
            final_loss = soft_min(all_group_losses, alpha=alpha)
        
        return final_loss
    
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluate the loss for a batch of atom positions."""
        self._debug_print("Starting loss evaluation")
        self._debug_print(f"Input positions shape: {positions.shape}")
        self._debug_print(f"Device: {positions.device}")
        
        # Pre-compute log probabilities for all resonance groups
        log_probs = []
        total_kde_evaluations = 0
        
        for group_idx, group in enumerate(self._resonance_groups.values()):
            self._debug_print(f"Processing resonance group {group_idx} with {len(group)} variants")
            group_log_probs = []
            
            for loss_idx, loss in enumerate(group):
                self._debug_print(f"   Evaluating variant {loss_idx}: {loss.method}")
                self._debug_print(f"     Weight: {loss._weight}, Offset: {loss._offset}")
                
                # Extract vertex positions
                vertex_positions = positions[:, loss._vertex_indices, :]
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                self._debug_print(f"     Vertex indices: {loss._vertex_indices}")
                self._debug_print(f"     Vertex positions shape: {vertex_positions.shape}")
                
                # Calculate distances
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                self._debug_print(f"     Distances shape: {dists.shape}")
                self._debug_print(f"     Distance stats: min={dists.min():.3f}, max={dists.max():.3f}, mean={dists.mean():.3f}")
                
                # Evaluate KDE
                logP = loss.kde_pdf.score_samples(dists)
                self._debug_print(f"     KDE log probabilities shape: {logP.shape}")
                self._debug_print(f"     LogP stats: min={logP.min():.3f}, max={logP.max():.3f}, mean={logP.mean():.3f}")
                
                group_log_probs.append(logP)
                total_kde_evaluations += 1
            
            # Pad group log_probs to match the maximum number of variants
            if len(group_log_probs) < self._resonance_groups_indices.shape[1]:
                padding_needed = self._resonance_groups_indices.shape[1] - len(group_log_probs)
                # Pad with zeros (these won't be used due to group sizes)
                for _ in range(padding_needed):
                    group_log_probs.append(torch.zeros_like(group_log_probs[0]))
                self._debug_print(f"   Padded group {group_idx} log_probs from {len(group)} to {len(group_log_probs)} variants")
            
            log_probs.append(torch.stack(group_log_probs))
        
        # Stack all log probabilities
        if log_probs:
            log_probs = torch.stack(log_probs)
            self._debug_print(f"Stacked log_probs shape: {log_probs.shape}")
        else:
            self._debug_print(f"No log probabilities computed - empty groups")
            return torch.zeros(positions.shape[0], device=positions.device)
        
        self._debug_print(f"Total KDE evaluations: {total_kde_evaluations}")
        self._debug_print(f"Calling compiled evaluation function...")
        self._debug_print(f"Pre-compiled inputs:")
        self._debug_print(f"   positions: {positions.shape}")
        self._debug_print(f"   indices: {self._resonance_groups_indices.shape}")
        self._debug_print(f"   weights: {self._resonance_groups_weights.shape}")
        self._debug_print(f"   offsets: {self._resonance_groups_offsets.shape}")
        self._debug_print(f"   log_probs: {log_probs.shape}")
        self._debug_print(f"   temp: {self._temp}, alpha: {self._alpha}")
        
        result = self._compiled_eval_loss(
            positions,
            self._resonance_groups_indices,
            self._resonance_groups_weights,
            self._resonance_groups_offsets,
            self._resonance_groups_sizes,
            self._temp,
            self._alpha,
            self._device,
            KB,
            log_probs
        )
        
        self._debug_print("Completed evaluation")
        self._debug_print(f"Result shape: {result.shape}")
        self._debug_print(f"Result stats: min={result.min():.3f}, max={result.max():.3f}, mean={result.mean():.3f}")
        
        return result
    
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
        self._debug_print("Starting loss initialization")
        
        if not self.strategies:
            raise ValueError(
                "No cyclization strategies provided and default_strategies is empty."
            )

        # Get the loaded trajectory
        traj = self.traj
        
        # Pre-compute atom indices for faster lookup
        self._debug_print(f"Pre-computing atom indices for {len(self.bonding_atoms)} atom types")
        atom_idx_dict = self.precompute_atom_indices(
            list(self.topology.residues),
            self.bonding_atoms
        )
        self._debug_print(f"Found {len(atom_idx_dict)} atom mappings")
        
        # Dictionary to group losses by resonance key
        resonance_groups: Dict[Tuple[str, frozenset], List[ChemicalLoss]] = {}
        total_losses_created = 0
        
        # Create all loss instances
        for strat_idx, strat in enumerate(self.strategies):
            self._debug_print(f"Processing strategy {strat_idx}: {strat.__name__}")
            strategy_losses = 0
            
            for pair_idx, idxs_method_pair in enumerate(strat.get_indexes_and_methods(traj, atom_idx_dict)):
                # Skip if either residue in the pair is masked
                if self._mask and (idxs_method_pair.pair & self._mask):
                    self._debug_print(f"   Skipping pair {pair_idx} (residues {sorted(idxs_method_pair.pair)}) - masked")
                    continue
                
                # Extract the base method (without resonance info) for grouping
                method_base = idxs_method_pair.method.split(" (")[0]
                resonance_key = (method_base, frozenset(idxs_method_pair.pair))
                
                self._debug_print(f"   Creating loss for pair {pair_idx}: {idxs_method_pair.method}")
                self._debug_print(f"     Residues: {sorted(idxs_method_pair.pair)}")
                self._debug_print(f"     Atom indices: {idxs_method_pair.indexes}")
                self._debug_print(f"     Weight: {self._weights[strat]}, Offset: {self._offsets[strat]}")
                
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
                    self._debug_print(f"     Created new resonance group: {method_base}")
                else:
                    self._debug_print(f"     Added to existing resonance group: {method_base}")
                    
                resonance_groups[resonance_key].append(loss)
                strategy_losses += 1
                total_losses_created += 1
            
            self._debug_print(f"Strategy {strat.__name__} created {strategy_losses} losses")
        
        # Store the grouped structure for efficient evaluation
        self._resonance_groups = resonance_groups
        
        # Also maintain a flat list for compatibility
        self._losses = []
        for group in resonance_groups.values():
            self._losses.extend(group)
        
        self._debug_print("Completed")
        self._debug_print(f"Total losses created: {total_losses_created}")
        self._debug_print(f"Total resonance groups: {len(resonance_groups)}")
        
        # Print resonance group summary
        for resonance_key, losses_in_group in resonance_groups.items():
            method_base, residue_pair = resonance_key
            self._debug_print(f"Group: {method_base} | residues={sorted(residue_pair)} | variants={len(losses_in_group)}")
    
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
    def strategies(self) -> Set[Type[ChemicalLoss]]:
        return self._strategies
    
    @property
    def weights(self) -> Dict[Type[ChemicalLoss], float]:
        return self._weights
    
    @property
    def offsets(self) -> Dict[Type[ChemicalLoss], float]:
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
    def losses(self) -> List[ChemicalLoss]:
        return self._losses
    
    @property
    def traj(self) -> md.Trajectory:
        return self._traj
    
    @property
    def topology(self) -> md.Topology:
        return self._topology 
    
    @property
    def summary(self) -> str:
        """
        Generate a concise but comprehensive summary of the ChemicalLossHandler state.
        Optimized for debugging while maintaining all critical information.
        """
        if not self._resonance_groups:
            return "No valid cyclization sites found."
        
        # Core configuration
        config = [
            f"CONFIG: pdb={self.pdb_path.name} | temp={self.temp}K | alpha={self.alpha} | device={self.device} | DEBUG=ON",
            f"UNITS: factor={self.units_factor}",
            f"STATS: total_losses={len(self._losses)} | groups={len(self._resonance_groups)} | strategies={len(self.strategies)}"
        ]
        
        # Strategy weights and offsets
        strategy_info = ["STRATEGIES:"]
        for strategy in sorted(self.strategies, key=lambda x: x.__name__):
            weight = self.weights.get(strategy, 1.0)
            offset = self.offsets.get(strategy, 0.0)
            count = sum(1 for loss in self._losses if isinstance(loss, strategy))
            strategy_info.append(f"  {strategy.__name__}: w={weight} o={offset} n={count}")
        
        # KDE models used
        kde_info = {}
        for loss in self._losses:
            kde_file = loss.kde_file
            if kde_file not in kde_info:
                kde_info[kde_file] = {'count': 0, 'strategies': set()}
            kde_info[kde_file]['count'] += 1
            kde_info[kde_file]['strategies'].add(type(loss).__name__)
        
        kde_summary = ["KDE MODELS:"]
        for kde_file, info in kde_info.items():
            kde_path = Path(kde_file)
            kde_summary.append(f"  {kde_path.name}: n={info['count']} | strategies={','.join(sorted(info['strategies']))}")
        
        # Resonance groups
        groups_info = ["RESONANCE GROUPS:"]
        for (method_base, residue_pair), losses_in_group in self._resonance_groups.items():
            is_resonant = len(losses_in_group) > 1
            group_type = "RESONANT" if is_resonant else "SINGLE"
            groups_info.append(f"  {group_type} | {method_base}")
            groups_info.append(f"    residues={sorted(residue_pair)} | variants={len(losses_in_group)}")
            
            # Add first variant details as reference
            first_loss = losses_in_group[0]
            atom_info = [f"{k}:{v}" for k, v in first_loss._atom_idxs.items()]
            groups_info.append(f"    atoms={','.join(atom_info)} | kde={Path(first_loss.kde_file).name}")
            if is_resonant:
                groups_info.append(f"    additional_variants={len(losses_in_group)-1}")
            groups_info.append("")
        
        # Combine all sections
        return "\n".join([
            *config,
            "",
            *strategy_info,
            "",
            *kde_summary,
            "",
            *groups_info
        ])

    def compute_individual_losses_log(self, positions: torch.Tensor) -> str: # we should change the name of this function
        """
        Compute and return a concise string showing individual loss values for logging.
        Reuses the core evaluation logic from _eval_loss but provides detailed breakdown.
        
        Args:
            positions: Tensor of shape [batch_size, n_atoms, 3]
            
        Returns:
            String with individual loss values, group minimums, and final loss for logging
        """
        if self._resonance_groups_indices.shape[0] == 0:
            return "No losses computed - empty resonance groups"
        
        # Apply units conversion just like __call__ method does
        positions_ang = positions * self.units_factor
        
        batch_size = positions_ang.shape[0]
        log_lines = [f"Loss breakdown for {batch_size} batch(es):"]
        
        # Pre-compute log probabilities for all resonance groups (reusing _eval_loss logic)
        log_probs = []
        group_losses_detailed = []
        resonance_keys_ordered = []  # Track the exact order we process groups
        
        for group_idx, (resonance_key, group) in enumerate(self._resonance_groups.items()):
            resonance_keys_ordered.append(resonance_key)  # Store the key for this position
            method_base, residue_pair = resonance_key
            group_log_probs = []
            group_individual_losses = []
            
            for loss_idx, loss in enumerate(group):
                # Extract vertex positions (same as _eval_loss)
                vertex_positions = positions_ang[:, loss._vertex_indices, :]
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                # Calculate distances (same as _eval_loss)
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                # Evaluate KDE (same as _eval_loss)
                logP = loss.kde_pdf.score_samples(dists)
                energy = -KB * self._temp * logP
                
                # Apply weight and offset
                scaled_energy = loss._weight * energy + loss._offset
                
                group_log_probs.append(logP)
                group_individual_losses.append({
                    'method': loss.method,
                    'residues': sorted(residue_pair),
                    'energy': scaled_energy,
                    'weight': loss._weight,
                    'offset': loss._offset
                })
            
            # Pad group log_probs to match tensor structure
            if len(group_log_probs) < self._resonance_groups_indices.shape[1]:
                padding_needed = self._resonance_groups_indices.shape[1] - len(group_log_probs)
                for _ in range(padding_needed):
                    group_log_probs.append(torch.zeros_like(group_log_probs[0]))
            
            log_probs.append(torch.stack(group_log_probs))
            group_losses_detailed.append(group_individual_losses)
        
        # Stack all log probabilities
        log_probs = torch.stack(log_probs) if log_probs else torch.empty((0, 0, batch_size), device=positions_ang.device)
        
        # Compute group minimums and add to log
        group_minimums = []
        for group_idx, group_losses in enumerate(group_losses_detailed):
            method_base = list(self._resonance_groups.keys())[group_idx][0]
            residue_pair = list(self._resonance_groups.keys())[group_idx][1]
            
            if len(group_losses) == 1:
                # Single variant
                loss_info = group_losses[0]
                group_min = loss_info['energy']
                log_lines.append(f"  Group {group_idx} ({method_base} | res={sorted(residue_pair)}): SINGLE variant")
                for batch_idx in range(batch_size):
                    log_lines.append(f"    Batch {batch_idx}: {loss_info['method']} = {group_min[batch_idx]:.4f}")
            else:
                # Multiple resonant variants
                log_lines.append(f"  Group {group_idx} ({method_base} | res={sorted(residue_pair)}): {len(group_losses)} resonant variants")
                
                energies = torch.stack([loss_info['energy'] for loss_info in group_losses], dim=1)
                group_min = torch.min(energies, dim=1)[0]
                
                for batch_idx in range(batch_size):
                    individual_values = [f"{loss_info['method'].split('(')[1].rstrip(')')}: {loss_info['energy'][batch_idx]:.4f}" 
                                       for loss_info in group_losses]
                    log_lines.append(f"    Batch {batch_idx}: [{' | '.join(individual_values)}] → min={group_min[batch_idx]:.4f}")
            
            group_minimums.append(group_min)
        
        # Compute final aggregated loss using compiled method
        final_loss = self._compiled_eval_loss(
            positions_ang,
            self._resonance_groups_indices,
            self._resonance_groups_weights,
            self._resonance_groups_offsets,
            self._resonance_groups_sizes,
            self._temp,
            self._alpha,
            self._device,
            KB,
            log_probs
        )
        
        # Add final aggregation info
        if len(group_minimums) > 1:
            log_lines.append(f"  Final aggregation (soft_min with α={self._alpha}):")
            
            # Add header showing what each column represents
            group_labels = []
            #resonance_keys_list = list(self._resonance_groups.keys())  # Store the exact order
            for group_idx, resonance_key in enumerate(resonance_keys_ordered):
                method_base, residue_pair = resonance_key
                group_labels.append(f"Col{group_idx}={method_base}|residues={sorted(residue_pair)}")
            header_line = "    " + " | ".join(group_labels)
            log_lines.append(header_line)
            log_lines.append("    " + "-" * len(header_line.strip()))
            
            for batch_idx in range(batch_size):
                group_mins_str = ", ".join([f"{group_min[batch_idx]:.4f}" for group_min in group_minimums])
                log_lines.append(f"    Batch {batch_idx}: [{group_mins_str}] → final={final_loss[batch_idx]:.4f}")
        else:
            log_lines.append(f"  Final loss (single group):")
            for batch_idx in range(batch_size):
                log_lines.append(f"    Batch {batch_idx}: {final_loss[batch_idx]:.4f}")
        
        # Add total sum for verification
        total_loss = torch.sum(final_loss).item()
        log_lines.append("")
        log_lines.append(f"TOTAL SUM (across all batches): {total_loss:.4f}")
        
        return "\n".join(log_lines)
    
    def get_strategy_losses_dict(self, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        Compute the minimum loss values for each resonance group and return a dictionary with the following structure:
        {
            'Col0=method_base|residues=[1,2]': torch.Tensor, # shape: [batch_size]
            'Col1=method_base|residues=[3,4]': torch.Tensor, # shape: [batch_size]
            ...
        }
        
        Args:
            positions: Tensor of shape [batch_size, n_atoms, 3]
            
        Returns:
            Dictionary mapping column names to group minimum tensors
        '''
        if self._resonance_groups_indices.shape[0] == 0:
            return {}
        
        # Apply units conversion just like __call__ method does
        positions_ang = positions * self.units_factor
        
        # Pre-compute log probabilities for all resonance groups (reusing _eval_loss logic)
        log_probs = []
        group_losses_detailed = []
        resonance_keys_ordered = []  # Track the exact order we process groups
        
        for group_idx, (resonance_key, group) in enumerate(self._resonance_groups.items()):
            resonance_keys_ordered.append(resonance_key)  # Store the key for this position
            method_base, residue_pair = resonance_key
            group_log_probs = []
            group_individual_losses = []
            
            for loss_idx, loss in enumerate(group):
                # Extract vertex positions (same as _eval_loss)
                vertex_positions = positions_ang[:, loss._vertex_indices, :]
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                # Calculate distances (same as _eval_loss)
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                # Evaluate KDE (same as _eval_loss)
                logP = loss.kde_pdf.score_samples(dists)
                energy = -KB * self._temp * logP
                
                # Apply weight and offset
                scaled_energy = loss._weight * energy + loss._offset
                
                group_log_probs.append(logP)
                group_individual_losses.append({
                    'method': loss.method,
                    'residues': sorted(residue_pair),
                    'energy': scaled_energy,
                    'weight': loss._weight,
                    'offset': loss._offset
                })
            
            # Pad group log_probs to match tensor structure
            if len(group_log_probs) < self._resonance_groups_indices.shape[1]:
                padding_needed = self._resonance_groups_indices.shape[1] - len(group_log_probs)
                for _ in range(padding_needed):
                    group_log_probs.append(torch.zeros_like(group_log_probs[0]))
            
            log_probs.append(torch.stack(group_log_probs))
            group_losses_detailed.append(group_individual_losses)
        
        # Stack all log probabilities
        log_probs = torch.stack(log_probs) if log_probs else torch.empty((0, 0, positions_ang.shape[0]), device=positions_ang.device)
        
        # Compute group minimums and create dictionary
        result_dict = {}
        for group_idx, group_losses in enumerate(group_losses_detailed):
            method_base = list(self._resonance_groups.keys())[group_idx][0]
            residue_pair = list(self._resonance_groups.keys())[group_idx][1]
            
            if len(group_losses) == 1:
                # Single variant
                loss_info = group_losses[0]
                group_min = loss_info['energy']
            else:
                # Multiple resonant variants
                energies = torch.stack([loss_info['energy'] for loss_info in group_losses], dim=1)
                group_min = torch.min(energies, dim=1)[0]
            
            # Create column name in the same format as compute_individual_losses_log
            col_name = f"Col{group_idx}={method_base}|residues={sorted(residue_pair)}"
            result_dict[col_name] = group_min
        
        return result_dict
        