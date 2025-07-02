"""
ChemicalLossHandler optimized for torch.jit compilation and batch processing.
This class is designed to be immutable after initialization.
"""

__all__ = ['ChemicalLossHandler']

from typing import Dict, List, Optional, Set, Type, Union, Tuple
import torch
import mdtraj as md
from pathlib import Path

from .loss_handler import LossHandler
from ..chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ...torchkde.modules import KernelDensity
from ...losses.utils.default_strategies import DEFAULT_STRATEGIES
from ...utils.utils import soft_min
from ...utils.constants import KB, ALLOWED_RESIDUES

# Type aliases for improved readability
PathLike = Union[str, Path]
ChemicalLossType = Type[ChemicalLoss]
StrategySet = Set[ChemicalLossType]
WeightDict = Dict[ChemicalLossType, float]
OffsetDict = Dict[ChemicalLossType, float]
MaskSet = Set[int]

class ChemicalLossHandler(LossHandler):
    """
    ChemicalLossHandler optimized for torch.jit compilation and batch processing.
    This class is designed to be immutable after initialization.
    """

    default_strategies: Set[Type[ChemicalLoss]] = DEFAULT_STRATEGIES
    
    def __init__(self,
                 traj: md.Trajectory,
                 units_factor: float,
                 strategies: Optional[StrategySet] = None,
                 weights: Optional[WeightDict] = None,
                 offsets: Optional[OffsetDict] = None,
                 temp: float = 1.0,
                 alpha: float = -3.0,
                 mask: Optional[MaskSet] = None,
                 device: Optional[torch.device] = None,
                 debug: bool = False,
                 ):
        super().__init__(units_factor)
        self._validate_chem_loss_handler_inputs(traj, strategies, weights, 
                                                offsets, temp, alpha, mask, 
                                                device, debug,
                                                )
        self._atom_indexes_dict = self._generate_atom_indexes_dict(traj.topology)
        self._initialize_resources(traj, strategies, weights, offsets, temp, alpha, mask, device)

        if len(self._chemical_losses) == 0:
            raise ValueError(
                "No chemical losses were created from the provided strategies and trajectory. "
                "This could happen if:\n"
                "1. The trajectory doesn't contain residues compatible with the selected strategies\n"
                "2. The selected strategies don't find any valid atom combinations\n"
                "3. All residues are filtered out by validation rules\n"
                "Please check your trajectory and strategy selection."
            )
        
    @classmethod
    def from_pdb_file(cls,
                      pdb_file: PathLike,
                      units_factor: float,
                      strategies: Optional[StrategySet] = None,
                      temp: float = 1.0,
                      alpha: float = -3.0,
                      mask: Optional[MaskSet] = None,
                      device: Optional[torch.device] = None,
                      debug: bool = False,
                      ) -> "ChemicalLossHandler":
        """
        Initialize a ChemicalLossHandler from a PDB file.
        """
        traj = md.load_pdb(pdb_file)
        return cls(traj, units_factor, strategies=strategies, temp=temp, alpha=alpha, mask=mask, device=device, debug=debug)

    def _validate_chem_loss_handler_inputs(
            self,
            traj: md.Trajectory,
            strategies: Optional[StrategySet],
            weights: Optional[WeightDict],
            offsets: Optional[OffsetDict],
            temp: float,
            alpha: float,
            mask: Optional[MaskSet],
            device: Optional[torch.device],
            debug: bool,
    ) -> None:
        """
        Validates the additional inputs to the ChemicalLossHandler.
        Don't need to validate the units_factor, because it is validated in the LossHandler.__init__ method.
        """
        def _validate_traj(traj: md.Trajectory) -> None:
            # TODO: check that AMBER CAPS are only at the N-terminus and C-terminus
            for residue in traj.topology.residues:
                if residue.name not in ALLOWED_RESIDUES:
                    raise ValueError(f"""Invalid residue name: {residue.name}. 
                                    Must be a canonical amino acid 3 letter code or an AMBER cap. 
                                    Valid codes are: {ALLOWED_RESIDUES}""")
            if not isinstance(traj, md.Trajectory):
                raise ValueError("traj must be an instance of mdtraj.Trajectory")
            if traj.n_atoms == 0:
                raise ValueError("traj must have at least one atom")
            if traj.n_residues == 0:
                raise ValueError("traj must have at least one residue")
            if traj.n_chains != 1: # check that trajectory has only one protein chain
                raise ValueError("traj must have only one chain")
        def _validate_strategies(strategies: Optional[StrategySet]) -> None:
            if strategies is not None:
                if not isinstance(strategies, set):
                    raise ValueError("strategies must be a set")
                for strategy in strategies:
                    if not (isinstance(strategy, type) and issubclass(strategy, ChemicalLoss)):
                        raise ValueError(f"All strategies must be subclasses of ChemicalLoss. Got {strategy}")
        def _validate_weights(weights: Optional[WeightDict]) -> None:
            if weights is not None:
                if not isinstance(weights, dict):
                    raise ValueError("weights must be a dictionary")
                for key, val in weights.items():
                    if val < 0.0:
                        raise ValueError(f"All provided weights must be positive. Got {val} for {key}")
        def _validate_offsets(offsets: Optional[OffsetDict]) -> None:
            if offsets is not None:
                if not isinstance(offsets, dict):
                    raise ValueError("offsets must be a dictionary")
                for key, val in offsets.items():
                    if not isinstance(val, (float, int)):
                        raise ValueError(f"All provided offsets must be floats. Got {type(val)} for {key}")
        def _validate_temp(temp: float) -> None:
            if temp <= 0.0:
                raise ValueError("temp must be positive")
        def _validate_alpha(alpha: float) -> None:
            if alpha >= 0.0:
                raise ValueError("alpha must be negative")
        def _validate_mask(mask: Optional[MaskSet]) -> None:
            if mask is not None:
                if not isinstance(mask, set):
                    raise ValueError("mask must be a set")
                for val in mask:
                    if not isinstance(val, int):
                        raise ValueError("All mask values must be integers")
        def _validate_device(device: Optional[torch.device]) -> None:
            if device is not None:
                if not isinstance(device, torch.device):
                    raise ValueError("device must be a torch.device")
        def _validate_debug(debug: bool) -> None:
            if not isinstance(debug, bool):
                raise ValueError("debug must be a boolean")
        # here we actually do our validation:
        _validate_debug(debug)
        _validate_traj(traj)
        _validate_strategies(strategies)
        _validate_weights(weights)
        _validate_offsets(offsets)
        _validate_temp(temp)
        _validate_alpha(alpha)
        _validate_mask(mask)
        _validate_device(device)

    def _initialize_chemical_losses(self) -> None:
        """
        Initialize the chemical losses for the ChemicalLossHandler, sorted 
        lexicographically by their resonance keys.
        """
        chemical_losses: List[ChemicalLoss] = []
        for loss_strategy in self._strategies:
            loss_instances = loss_strategy.get_loss_instances(traj=self._traj, 
                                                              atom_indexes_dict=self.atom_indexes_dict,
                                                              weight=self._weights[loss_strategy],
                                                              offset=self._offsets[loss_strategy],
                                                              temp=self._temp,
                                                              device=self._device,
                                                              )
            chemical_losses.extend(loss_instances)
        # Check for None resonance keys and raise error if found
        for loss in chemical_losses:
            if loss.resonance_key is None:
                raise ValueError(f"Resonance key cannot be None for loss {loss} in a ChemicalLossHandler")
        # Remove any losses which have atoms in masked amino acids
        if self._mask is not None:
            chemical_losses = [loss for loss in chemical_losses if not any(residue_idx in self._mask for residue_idx in loss.resonance_key[1])]
        # Sort by resonance key: first by string, then by sorted frozenset
        chemical_losses.sort(key=lambda loss: (
            loss.resonance_key[0],
            tuple(sorted(loss.resonance_key[1]))
        ))
        self._chemical_losses: Tuple[ChemicalLoss, ...] = tuple(chemical_losses)
    
    def _initialize_kde_groups(self) -> None:
        """
        Initialize the KDE groups for the ChemicalLossHandler.

        Makes self._kde_groups to be Dict[KernelDensity, Tuple[torch.Tensor, torch.Tensor]]
        First tensor is of shape [n_loss_subset, ] and contains which indexes in the chemical losses are in that kde group.
        The next tensor is of shape [n_loss_subset, 4] and corresponds to the 4 points in the protein that each loss
        is evaluated on.
        """
        kde_groups: Dict[KernelDensity, List[torch.Tensor]] = {}

        for i in range(self.n_losses):
            chem_loss = self._chemical_losses[i]
            current_kde = chem_loss.kde_pdf
            if current_kde not in kde_groups:
                kde_groups[current_kde] = [torch.tensor([i], dtype=torch.long, device=self._device), chem_loss.vertex_indices.unsqueeze(0)]
            else:
                kde_groups[current_kde][0] = torch.cat((kde_groups[current_kde][0], torch.tensor([i], dtype=torch.long, device=self._device))) # shape [n_loss_subset, ]
                kde_groups[current_kde][1] = torch.cat((kde_groups[current_kde][1], chem_loss.vertex_indices.unsqueeze(0)), dim=0) # shape [n_loss_subset, 4] # shape [n_loss_subset, 4]
        
        # Make tensors contiguous in memory and store as instance variable
        self._kde_groups = {
            kde: (indices.contiguous(), vertices.contiguous()) 
            for kde, (indices, vertices) in kde_groups.items()
        }
    
    def _initialize_resonance_groups(self) -> None:
        """
        Initialize the resonance groups for the ChemicalLossHandler.
        Makes self._resonance_groups to be a tensor of shape [n_losses, ] and contains which group each
        loss belongs to.
        """
        resonance_groups = torch.zeros(len(self._chemical_losses), dtype=torch.long, device=self._device)
        
        tag = -1
        prev_key = None

        for i in range(self.n_losses):
            resonance_key = self._chemical_losses[i].resonance_key
            if resonance_key != prev_key:
                tag += 1
                prev_key = resonance_key
            resonance_groups[i] = tag
        self._resonance_groups = resonance_groups

    def _initialize_weight_and_offset_tensors(self) -> None:
        """
        Initialize the weight and offset tensors for the ChemicalLossHandler.
        Makes self._weight_tensor and self._offset_tensor to be tensors of shape [n_resonance_groups, ]
        and contains the weight and offset for each resonance group.

        Also makes self._full_weight_tensor and self._full_offset_tensor to be tensors of shape [n_losses, ]
        and contains the weight and offset for each loss. This does not include resonance groups.
        """
        weight_tensor = torch.zeros(self.n_resonance_groups, 
                                    dtype=torch.float, 
                                    device=self._device) # shape [n_resonance_groups, ]
        offset_tensor = torch.zeros(self.n_resonance_groups, 
                                    dtype=torch.float, 
                                    device=self._device) # shape [n_resonance_groups, ]
        
        for i in range(self.n_losses):
            loss = self._chemical_losses[i]
            group_idx = self._resonance_groups[i]
            weight_tensor[group_idx] = self._weights[type(loss)]
            offset_tensor[group_idx] = self._offsets[type(loss)]
        
        self._weight_tensor = weight_tensor # shape [n_resonance_groups, ]
        self._offset_tensor = offset_tensor # shape [n_resonance_groups, ]

        # Make full weight and offset tensors
        full_weight_tensor = torch.zeros(self.n_losses, 
                                         dtype=torch.float, 
                                         device=self._device) # shape [n_losses, ]
        full_offset_tensor = torch.zeros(self.n_losses, 
                                         dtype=torch.float, 
                                         device=self._device) # shape [n_losses, ]
        for i in range(self.n_losses):
            loss = self._chemical_losses[i]
            full_weight_tensor[i] = self._weights[type(loss)]
            full_offset_tensor[i] = self._offsets[type(loss)]
        
        self._full_weight_tensor = full_weight_tensor # shape [n_losses, ]
        self._full_offset_tensor = full_offset_tensor # shape [n_losses, ]

    def _initialize_resources(self,
                             traj: md.Trajectory,
                             strategies: Optional[StrategySet],
                             weights: Optional[WeightDict],
                             offsets: Optional[OffsetDict],
                             temp: float,
                             alpha: float,
                             mask: Optional[MaskSet],
                             device: Optional[torch.device]) -> None:
        """
        Initialize the resources for the ChemicalLossHandler.
        """
        # direct initializations
        self._traj = traj
        self._strategies = strategies or self.default_strategies
        self._weights = weights or {s: 1.0 for s in self._strategies}
        self._offsets = offsets or {s: 0.0 for s in self._strategies}
        self._temp = float(temp)
        self._alpha = float(alpha)
        self._mask = mask
        self._device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # method based initializations
        self._initialize_chemical_losses()
        self._initialize_kde_groups()
        self._initialize_resonance_groups()
        self._initialize_weight_and_offset_tensors()
        self._initialize_distance_optimization_structures()

    def _initialize_distance_optimization_structures(self) -> None:
        """
        Initialize structures for parallelized distance computation.
        Creates flattened vertex tensor and mapping info for efficient processing.
        """
        # Collect all vertices and track KDE group boundaries
        all_vertices = []  # Will hold vertex tensors from each KDE group
        kde_start_indices = []  # Starting index in flattened arrays for each KDE group
        kde_lengths = []  # Number of losses in each KDE group
        
        current_offset = 0
        for kde, (indices, vertices) in self._kde_groups.items():
            # vertices: shape [n_losses_in_group, 4] - atom indices for this KDE group
            all_vertices.append(vertices)
            
            # Track where this KDE group starts in the flattened tensor
            kde_start_indices.append(current_offset)
            kde_lengths.append(len(indices))
            
            current_offset += len(indices)
        
        # Create flattened tensors for distance computation optimization
        # _all_vertices: [total_losses, 4] - All vertex indices flattened into one tensor
        # Row i contains the 4 atom indices that define the tetrahedron for loss i
        self._all_vertices = torch.cat(all_vertices, dim=0)
        
        # _kde_start_indices: [n_kde_groups] - Starting position of each KDE group in flattened arrays
        # kde_start_indices[k] = first index where KDE group k's data begins
        self._kde_start_indices = torch.tensor(kde_start_indices, dtype=torch.long, device=self._device)
        
        # _kde_lengths: [n_kde_groups] - Number of losses in each KDE group
        # kde_lengths[k] = how many losses belong to KDE group k
        self._kde_lengths = torch.tensor(kde_lengths, dtype=torch.long, device=self._device)
        
        # Create ordered list of KDE objects for consistent indexing
        # _kde_list: List[KernelDensity] - Ordered list of KDE objects for consistent indexing
        # kde_list[k] = the actual KDE object for group k (matches kde_start_indices indexing)
        self._kde_list = list(self._kde_groups.keys())

    @property
    def topology(self) -> md.Topology:
        return self._traj.topology
    @staticmethod
    def _generate_atom_indexes_dict(topology: md.Topology) -> AtomIndexDict:
        indices = {}
        for residue in topology.residues:
            for atom in residue.atoms:
                indices[(residue.index, atom.name)] = atom.index
        return indices
    @property
    def atom_indexes_dict(self) -> AtomIndexDict:
        """Precompute a dictionary mapping (residue_index, atom_name) to atom indices."""
        return self._atom_indexes_dict
    @property
    def n_resonance_groups(self) -> int:
        if self.n_losses == 0:
            return 0
        return int(torch.max(self._resonance_groups) + 1)
    @property
    def n_losses(self) -> int:
        return len(self._chemical_losses)
    @property
    def chemical_losses(self) -> Tuple[ChemicalLoss, ...]:
        """
        Immutable tuple of ChemicalLoss objects. Serves as a read-only view of 
        the `ChemicalLoss` objects which power the `ChemicalLossHandler` instance.
        """
        return self._chemical_losses
    
    @staticmethod
    @torch.jit.script # TODO: check this!!!
    def _compute_distances(
        positions: torch.Tensor, # shape [n_batch, n_atoms, 3], all floats
        vertexes: torch.Tensor, # shape [n_variations, 4], all ints in range (n_atoms), all longs
        ) -> torch.Tensor: # shape [n_batch, n_variations, 6]
        """
        Compute the distances between the vertices of the relevant 
        tetrahedron given all atom positions
        """
        
        # Extract vertex positions for all variations at once
        # Shape: [n_batch, n_variations, 4, 3]
        vertex_positions = positions[:, vertexes, :]  # Broadcasting: [n_batch, 1, n_atoms, 3] with [n_variations, 4]
        
        # Extract individual vertices
        # Each has shape: [n_batch, n_variations, 3]
        v0 = vertex_positions[:, :, 0, :]
        v1 = vertex_positions[:, :, 1, :]
        v2 = vertex_positions[:, :, 2, :]
        v3 = vertex_positions[:, :, 3, :]
        
        # Stack atom pairs for vectorized distance calculation
        # Each stack operation results in shape: [n_batch, n_variations, 6, 3]
        atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=2)
        atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=2)
        
        # Calculate pairwise distances
        # Shape: [n_batch, n_variations, 6]
        dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
        
        return dists # shape [n_batch, n_variations, 6]

    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        PROVIDED POSITIONS ARE IN ANGSTROMS.
        Evaluate the loss for a batch of atom positions via the following steps:
        1. Compute ALL distances at once (parallelized optimization)
        2. For each KDE group, slice relevant distances and evaluate KDE PDF
        3. Take the minimum value of the KDE PDF over each resonance group
        4. Apply the weight and offset tensors to the minimum values
        5. Take a soft minimum across all losses
        6. Return a loss of shape [n_batch, ]
        """
        n_batch = positions.shape[0]
        raw_losses = torch.zeros(n_batch, self.n_losses, dtype=torch.float, device=self._device) # shape [n_batch, n_losses]
        
        # Step 1: Compute ALL distances at once for efficiency (NEW OPTIMIZATION)
        # This replaces individual _compute_distances calls inside the loop
        all_distances = self._compute_distances(positions, self._all_vertices) # shape [n_batch, total_losses, 6]
        
        # Step 2: For each KDE group, slice distances and evaluate PDF (FUNCTIONALITY UNCHANGED)
        for kde_idx, kde in enumerate(self._kde_list):
            # Get the original chemical loss indices for this KDE group (for storing results)
            original_indices, _ = self._kde_groups[kde] # [n_loss_subset, ]
            
            # Slice the pre-computed distances for this KDE group
            start_idx = self._kde_start_indices[kde_idx]
            length = self._kde_lengths[kde_idx]
            end_idx = start_idx + length
            distances = all_distances[:, start_idx:end_idx, :] # shape [n_batch, n_loss_subset, 6]
            
            # Rest is IDENTICAL to original implementation
            # Evaluate KDE PDF at the distances
            # Reshape to [n_batch * n_loss_subset, 6] for KDE evaluation
            n_loss_subset = distances.shape[1]
            distances_flat = distances.view(-1, 6) # shape [n_batch * n_loss_subset, 6]
            
            # Evaluate KDE and reshape back
            kde_values_flat = kde.score_samples(distances_flat) # shape [n_batch * n_loss_subset, ]
            kde_values = kde_values_flat.view(n_batch, n_loss_subset) # shape [n_batch, n_loss_subset]
            
            # Store results in raw_losses at the appropriate indices
            raw_losses[:, original_indices] = kde_values # shape [n_batch, n_loss_subset] -> [n_batch, n_losses] at indices
        
        # Step 3: Take minimum value over each resonance group
        resonance_losses = torch.zeros(n_batch, self.n_resonance_groups, dtype=torch.float, device=self._device) # shape [n_batch, n_resonance_groups]
        
        for group_idx in range(self.n_resonance_groups):
            # Find all losses belonging to this resonance group
            group_mask = (self._resonance_groups == group_idx) # shape [n_losses, ]
            group_losses = raw_losses[:, group_mask] # shape [n_batch, n_group_members]
            
            # Take minimum over the group
            resonance_losses[:, group_idx] = torch.min(group_losses, dim=1)[0] # shape [n_batch, ]
        
        # Step 3.5: Apply temp and KB to the resonance losses
        tempered_losses = -KB * self._temp * resonance_losses

        # Step 4: Apply weight and offset tensors 
        weighted_losses = self._weight_tensor.unsqueeze(0) * tempered_losses + self._offset_tensor.unsqueeze(0) # shape [n_batch, n_resonance_groups]
        
        # Step 5: Sum to get final loss
        final_loss = soft_min(weighted_losses, alpha=self._alpha) # shape [n_batch, ]
        
        return final_loss
    
    def _eval_loss_explicit(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Directly computes the loss, without exploiting the parallelization of the loss computation.
        This will be much slower than the optimized version, but it is useful for debugging and testing.
        """
        n_batch = positions.shape[0]
        losses = torch.zeros(n_batch, self.n_losses, dtype=torch.float, device=self._device) # shape [n_batch, n_losses]
        
        for i in range(len(self._chemical_losses)):
            loss = self._chemical_losses[i]
            loss_values = loss(positions)
            losses[:, i] = loss_values
        
        # Step 3: Take minimum value over each resonance group
        resonance_losses = torch.zeros(n_batch, self.n_resonance_groups, dtype=torch.float, device=self._device) # shape [n_batch, n_resonance_groups]
        
        for group_idx in range(self.n_resonance_groups):
            # Find all losses belonging to this resonance group
            group_mask = (self._resonance_groups == group_idx) # shape [n_losses, ]
            group_losses = losses[:, group_mask] # shape [n_batch, n_group_members]
            
            # Take minimum over the group
            resonance_losses[:, group_idx] = torch.min(group_losses, dim=1)[0] # shape [n_batch, ]
        
        # Step 5: Take a soft minimum across all losses
        final_loss = soft_min(resonance_losses, alpha=self._alpha) # shape [n_batch, ]
        
        return final_loss
            
    def _call_explicit(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Call the explicit loss evaluation method. Should be identical to __call__ method (but much slower)
        """
        return self._eval_loss_explicit(positions * self._units_factor)
    
    def _get_smallest_loss_index(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get the index of the smallest loss per batch.
        Returns a tensor of shape [n_batch, ] and contains the index of the smallest loss for each batch.
        """
        n_batch = positions.shape[0]
        raw_losses = torch.zeros(n_batch, self.n_losses, dtype=torch.float, device=self._device) # shape [n_batch, n_losses]
        
        # Step 1: Compute ALL distances at once for efficiency (NEW OPTIMIZATION)
        # This replaces individual _compute_distances calls inside the loop
        all_distances = self._compute_distances(positions, self._all_vertices) # shape [n_batch, total_losses, 6]
        
        # Step 2: For each KDE group, slice distances and evaluate PDF (FUNCTIONALITY UNCHANGED)
        for kde_idx, kde in enumerate(self._kde_list):
            # Get the original indices for this KDE group (for storing results)
            original_indices, _ = self._kde_groups[kde]
            
            # Slice the pre-computed distances for this KDE group
            start_idx = self._kde_start_indices[kde_idx]
            length = self._kde_lengths[kde_idx]
            end_idx = start_idx + length
            distances = all_distances[:, start_idx:end_idx, :] # shape [n_batch, n_loss_subset, 6]
            
            # Rest is IDENTICAL to original implementation
            # Evaluate KDE PDF at the distances
            # Reshape to [n_batch * n_loss_subset, 6] for KDE evaluation
            n_loss_subset = distances.shape[1]
            distances_flat = distances.view(-1, 6) # shape [n_batch * n_loss_subset, 6]
            
            # Evaluate KDE and reshape back
            kde_values_flat = kde.score_samples(distances_flat) # shape [n_batch * n_loss_subset, ]
            kde_values = kde_values_flat.view(n_batch, n_loss_subset) # shape [n_batch, n_loss_subset]
            
            # Store results in raw_losses at the appropriate indices
            raw_losses[:, original_indices] = kde_values # shape [n_batch, n_loss_subset] -> [n_batch, n_losses] at indices

        # Step 3.5: Apply temp and KB to the raw losses
        tempered_losses = -KB * self._temp * raw_losses
        
        weighted_losses = tempered_losses * self._full_weight_tensor.unsqueeze(0) + self._full_offset_tensor.unsqueeze(0) # shape [n_batch, n_losses]
        
        # Find the index of the smallest loss for each batch
        smallest_loss_indices = torch.argmin(weighted_losses, dim=1) # shape [n_batch, ]
        return smallest_loss_indices
    
    def _get_smallest_loss(self, positions: torch.Tensor) -> Tuple[ChemicalLoss, ...]:
        """
        Get the smallest loss for each batch.
        Returns a tuple of ChemicalLoss objects.
        """
        smallest_loss_indices = self._get_smallest_loss_index(positions)
        return tuple(self._chemical_losses[i] for i in smallest_loss_indices)
    
    @property
    def _raw_summary(self, ) -> str:
        """
        Returns a string which summarizes the constituant ChemicalLoss objects
        """
        if self.n_losses == 0:
            return ""
        
        summary: List[str] = []

        for loss in self._chemical_losses:
            resonance_key = loss._resonance_key
            method = resonance_key[0]
            idxs = list(resonance_key[1])
            idxs.sort()
            summary.append(method + " " + str(idxs))

        final_str = ''
        for string in summary:
            final_str = final_str + string + ", "

        return final_str.strip()
    
    @property
    def _grouped_summary(self, ) -> str:
        """
        Returns a string which summarizes the constituent ChemicalLoss objects
        grouped by resonance groups, with counts for multiple losses in the same group.
        """
        if self.n_losses == 0:
            return ""
        
        # Group losses by resonance group
        resonance_groups = self._resonance_groups
        group_summaries = {}
        
        for i in range(self.n_losses):
            group_idx = int(resonance_groups[i])
            loss = self._chemical_losses[i]
            resonance_key = loss._resonance_key
            method = resonance_key[0]
            idxs = list(resonance_key[1])
            idxs.sort()
            loss_string = method + " " + str(idxs)
            
            if group_idx not in group_summaries:
                group_summaries[group_idx] = {
                    'string': loss_string,
                    'count': 1
                }
            else:
                group_summaries[group_idx]['count'] += 1
        
        # Build the final summary string
        summary_parts = []
        for group_idx in sorted(group_summaries.keys()):
            group_info = group_summaries[group_idx]
            loss_string = group_info['string']
            count = group_info['count']
            
            if count > 1:
                summary_parts.append(f"{loss_string} (x{count} resonance groups)")
            else:
                summary_parts.append(loss_string)
        
        return ", ".join(summary_parts)

    @property
    def summary(self, ) -> str:
        """
        Returns a string which summarizes the constituent ChemicalLoss objects
        grouped by resonance groups, with counts for multiple losses in the same group.
        """
        #header = "Constituent Losses:\n"
        #info = self._grouped_summary
        #bar = len(info) * "-" + "\n"
        #return header + bar + info
        return self._grouped_summary