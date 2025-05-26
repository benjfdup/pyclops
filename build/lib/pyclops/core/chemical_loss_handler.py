from typing import Dict, List, Optional, Set, Type, Union, Tuple, Any
import re

import torch
import mdtraj as md

from pathlib import Path
from ..core.chemical_loss import ChemicalLoss
from ..core.loss_handler import LossHandler
from ..utils.constants import KB, UNITS_FACTORS_DICT
from ..utils.default_strategies import DEFAULT_STRATEGIES
from ..utils.utils import soft_min


class ChemicalLossHandler(LossHandler):
    """
    A handler for chemical loss functions that evaluates the energetic
    favorability of different cyclization chemistries in peptide structures.
    
    This class automatically identifies potential cyclization sites in a given
    PDB structure and evaluates losses for each possible cyclization chemistry.
    The final loss combines all possible cyclizations using a soft minimum for 
    differentiable optimization.
    
    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file containing the peptide structure
    
    units_factor : float
        Input coordinate units.
    
    strategies : Set[Type[ChemicalLoss]], optional
        Set of chemical loss types to consider. If None, uses the default set.
    
    weights : Dict[Type[ChemicalLoss], float], optional
        Dictionary mapping loss types to their weights. Default is 1.0 for all strategies.
    
    offsets : Dict[Type[ChemicalLoss], float], optional
        Dictionary mapping loss types to constant offsets. Default is 0.0 for all strategies.
    
    temp : float, optional
        Temperature in Kelvin for Boltzmann weighting. Default is 300.0K.
    
    alpha : float, optional
        Soft minimum parameter. More negative values approach hard minimum,
        values near zero approach averaging. Default is -3.0.
    
    device : torch.device, optional
        Device to run calculations on. Default is CUDA if available, else CPU.
    
    Attributes
    ----------
    losses : List[ChemicalLoss]
        List of all instantiated loss functions.
    
    Methods
    -------
    __call__(positions)
        Evaluate the loss for a batch of atom positions.
    
    get_smallest_loss(positions)
        Get the loss with the smallest value for each structure in the batch.
    
    get_smallest_loss_methods(positions)
        Get the method names of the smallest losses for each structure.
    
    summary()
        Generate a human-readable summary of all detected cyclization options.
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
                temp: float = 300.0,
                alpha: float = -3.0,
                device: Optional[torch.device] = None,
                **kwargs) -> "ChemicalLossHandler":
        """
        Create a ChemicalLossHandler from a PDB file with simplified parameters.
        
        This is the recommended way to initialize a ChemicalLossHandler.
        
        Parameters
        ----------
        pdb_path : str or Path
            Path to the PDB file containing the peptide structure
        
        units : str, optional
            The units of the input coordinates. Must be one of: 'angstrom', 'nanometer', or 'nm'.
            If provided, `units_factor` should not be specified.

        units_factor : float, optional
            Direct conversion factor from input units to angstroms:
            `x_input * units_factor -> x_output (in angstroms)`
            If provided, `units` should not be specified.
        
        temp : float, optional
            Temperature in Kelvin for Boltzmann weighting. Default is 300.0K.
        
        alpha : float, optional
            Soft minimum parameter. More negative values approach hard minimum,
            values near zero approach averaging. Default is -3.0.
        
        device : torch.device, optional
            Device to run calculations on. Default is CUDA if available, else CPU.
        
        **kwargs : dict
            Additional keyword arguments to pass to the constructor.
        
        Returns
        -------
        ChemicalLossHandler
            A configured handler ready to evaluate losses.

        Examples
        --------
        >>> # Using named units
        >>> handler1 = ChemicalLossHandler.from_pdb("my_peptide.pdb", units="nanometer")
        >>> 
        >>> # Using direct conversion factor
        >>> handler2 = ChemicalLossHandler.from_pdb("my_peptide.pdb", units_factor=10.0)
        >>> 
        >>> # Using with positions
        >>> positions = torch.randn(10, 100, 3)  # 10 structures, 100 atoms, 3D
        >>> loss = handler1(positions)
        
        Notes
        -----
        You must provide either `units` or `units_factor`, but not both.
        """
        # Validate and determine units_factor
        if units is not None and units_factor is not None:
            raise ValueError(
                "Provide either 'units' or 'units_factor', but not both. "
                "This avoids ambiguity in the coordinate conversion."
            )
        
        if units is None and units_factor is None:
            raise ValueError(
                "Either 'units' or 'units_factor' must be provided to "
                "properly convert input coordinates to angstroms."
            )
        
        # Get the conversion factor from unit name
        if units is not None:
            try:
                units_factor = cls.units_factors_dict[units]
            except KeyError:
                valid_units = ', '.join(f"'{u}'" for u in cls.units_factors_dict.keys())
                raise ValueError(
                    f"Unknown unit: '{units}'. Valid units are: {valid_units}"
                )
        
        # Initialize the handler
        return cls(
            pdb_path=pdb_path,
            units_factor=units_factor,
            temp=temp,
            alpha=alpha,
            device=device,
            **kwargs
        )

    def __init__(self,
                 pdb_path: Union[str, Path],
                 units_factor: float,
                 strategies: Optional[Set[Type[ChemicalLoss]]] = None,
                 weights: Optional[Dict[Type[ChemicalLoss], float]] = None,
                 offsets: Optional[Dict[Type[ChemicalLoss], float]] = None,
                 temp: float = 300.0,
                 alpha: float = -3.0,
                 device: Optional[torch.device] = None):
        """
        Initialize a ChemicalLossHandler with detailed control over parameters.
        
        For most use cases, the `from_pdb` class method provides a simpler interface.
        """
        # Validate and process parameters
        
        super().__init__(units_factor)
        self._pdb_path = Path(pdb_path)
        
        if not self._pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {self._pdb_path}")
        
        # Load all strategy classes from the core.chemical_loss module if not provided
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
        
        # Initialize losses and optimization structures
        self._initialize_losses()
        self._build_kde_groups()
        
        # Set up resonance group tracking for efficient computation
        self._setup_resonance_groups()

        if not self._group_idx_tensor.max() < self._n_groups:
            raise ValueError("The group indices of the instantiated `ChemicalLossHandler` exceed number of groups")
        
    @property
    def pdb_path(self) -> Path:
        """Path to the PDB file."""
        return self._pdb_path
    
    @property
    def units_factor(self) -> float:
        """Conversion factor to Angstroms."""
        return self._units_factor
    
    @property
    def strategies(self) -> Set[Type[ChemicalLoss]]:
        """Set of chemical loss strategies."""
        return self._strategies
    
    @property
    def weights(self) -> Dict[Type[ChemicalLoss], float]:
        """Dictionary mapping loss types to weights."""
        return self._weights
    
    @property
    def offsets(self) -> Dict[Type[ChemicalLoss], float]:
        """Dictionary mapping loss types to offsets."""
        return self._offsets
    
    @property
    def alpha(self) -> float:
        """Soft minimum parameter."""
        return self._alpha
    
    @property
    def temp(self) -> float:
        """Temperature in Kelvin."""
        return self._temp
    
    @property
    def device(self) -> torch.device:
        """Computation device."""
        return self._device
    
    @property
    def losses(self) -> List[ChemicalLoss]:
        """List of instantiated loss functions."""
        return self._losses
    
    @property
    def traj(self) -> md.Trajectory:
        """MDTraj trajectory object."""
        return self._traj
    
    @property
    def topology(self) -> md.Topology:
        """MDTraj topology object."""
        return self._topology
    
    ### initialization methods ###

    @staticmethod
    def precompute_atom_indices(residues, atom_names) -> Dict[Tuple[int, str], int]:
        """
        Precompute a dictionary mapping (residue_index, atom_name) to atom indices.
        
        Parameters
        ----------
        residues : iterable
            Iterable of MDTraj Residue objects.
        
        atom_names : iterable
            Set of atom names to include.
        
        Returns
        -------
        Dict[Tuple[int, str], int]
            Dictionary mapping (residue_index, atom_name) to atom index.
        """
        indices = {}
        for residue in residues:
            for name in atom_names:
                atom = next((a for a in residue.atoms if a.name == name), None)
                if atom:
                    indices[(residue.index, name)] = atom.index
        return indices
    
    def _initialize_losses(self) -> None:
        """
        Initialize all loss functions by identifying valid cyclization sites.
        
        This method:
        1. Extracts atom indices from the PDB structure
        2. Identifies all potential cyclization sites for each strategy
        3. Creates loss function instances for each valid site
        4. Groups resonant structures for efficient computation
        """
        if not self.strategies: # checks if this is an empty set.
            raise ValueError(
                "No cyclization strategies provided and default_strategies is empty. "
                "The ChemicalLossHandler requires at least one strategy to function. "
                "Please provide strategies when initializing or implement the default_strategies."
            )

        # Get the loaded trajectory from the class
        traj = self.traj
        
        # Pre-compute a dictionary mapping (residue_index, atom_name) -> atom_index
        # for faster atom lookups during cyclization site identification
        atom_idx_dict = self.precompute_atom_indices(
            list(self.topology.residues),  # List of all residues in the structure
            self.bonding_atoms             # Set of atom names that participate in cyclization
        )
        
        # List to store all created loss function instances
        losses: List[ChemicalLoss] = []
        
        # === Resonance tracking system ===
        # Resonance occurs when multiple loss instances represent the same chemical interaction
        # but with different atom choices (e.g., due to carboxylate resonance)
        
        # List to store group ID for each loss (will be converted to tensor later)
        _resonance_group_ids = []
        
        # Map from resonance key to group ID
        # Key: (method_string, frozenset(residue_indices))
        # Value: unique integer ID for the resonance group
        resonance_key_to_id = {}
        group_counter = 0  # Counter for assigning unique group IDs
        
        # Iterate through all cyclization strategies (e.g., Disulfide, AmideLysTail, etc.)
        for strat in self.strategies:
            # For each strategy, identify all valid cyclization sites in the structure
            # This returns a list of IndexesMethodPair objects containing:
            # - indexes: Dict mapping atom keys to atom indices
            # - method: String describing the specific cyclization (e.g., "Disulfide, CYS 3 -> CYS 14")
            # - pair: Set of residue indices involved in this cyclization
            for idxs_method_pair in strat.get_indexes_and_methods(traj, atom_idx_dict):
                
                # Create the resonance key for the loss
                key = (idxs_method_pair.method, frozenset(idxs_method_pair.pair))

                # Create a loss function instance for this cyclization site
                loss = strat(
                    method=idxs_method_pair.method,        # Descriptive string for this cyclization
                    atom_idxs=idxs_method_pair.indexes,    # Atom indices for the tetrahedron
                    temp=self.temp,                        # Temperature for Boltzmann weighting
                    weight=self.weights[strat],            # Weight for this strategy type
                    offset=self.offsets[strat],            # Offset for this strategy type

                    # === Resonance tracking ===
                    # Create a unique key for this loss that identifies its resonance group
                    # The key combines the method string and the set of residues involved
                    # This ensures that different atom choices for the same chemical interaction
                    # (e.g., OE1 vs OE2 in glutamate) are grouped together
                    
                    # Store the resonance key in the loss instance for later reference
                    resonance_key= key,
                    device=self.device, # Computation device
                )
                
                # If this is the first time seeing this resonance key, assign a new group ID
                if key not in resonance_key_to_id:
                    resonance_key_to_id[key] = group_counter
                    group_counter += 1
                
                # Get the group ID for this resonance key
                group_id = resonance_key_to_id[key]
                
                # Track the group ID and add the loss to our collection
                _resonance_group_ids.append(group_id)
                losses.append(loss)
        
        # Convert the list of group IDs to a tensor for efficient operations later
        # This tensor parallel the losses list, providing the resonance group ID
        # for each loss instance
        self._resonance_group_ids = torch.tensor(
            _resonance_group_ids, 
            device=self.device, 
            dtype=torch.long
        )
        
        # Store the list of loss instances
        self._losses: List[ChemicalLoss] = losses
    
    def _build_kde_groups(self):
        """
        Group losses by their KDE functions for efficient batch evaluation.
        """
        kde_groups: Dict[Any, List[ChemicalLoss]] = {}  # Dictionary mapping KDE functions to lists of loss objects
        
        for loss in self._losses:
            kde = loss.kde_pdf
            if kde not in kde_groups:  # Check if we have an entry for this KDE
                kde_groups[kde] = []  # Initialize an empty list for this KDE
            kde_groups[kde].append(loss)  # Add the loss to this KDE's list
        
        final_groups: List[Dict[str, Any]] = []
        for kde, group in kde_groups.items():
            vertex_indices = torch.stack([l.get_vertexes_atomic_idxs() for l in group]) # shape: [n_loss, 4]
            weights = torch.tensor([l.weight for l in group], device=self.device)
            offsets = torch.tensor([l.offset for l in group], device=self.device)
            
            final_groups.append({
                "kde": kde,
                "losses": group,
                "atom_indices": vertex_indices, # [n_loss, 4]
                "weights": weights,
                "offsets": offsets,
            })
        
        self._kde_groups = final_groups
    
    def _setup_resonance_groups(self) -> None:
        """Set up structures for efficient handling of resonance groups."""
        group_ids = self._resonance_group_ids
        unique_ids = torch.unique(group_ids)
        group_id_map = {gid.item(): i for i, gid in enumerate(unique_ids)}
        
        self._group_idx_tensor: torch.tensor = torch.tensor(
            [group_id_map[i.item()] for i in group_ids], 
            device=self.device, 
            dtype=torch.long
        )
        self._n_groups = len(unique_ids)
    
    ### initialization methods ^^^ ###
    
    @staticmethod
    def _get_batch_geometry(positions: torch.tensor) -> torch.tensor:
        """
        Compute pairwise distances for batches of tetrahedral geometries.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_loss, 4, 3) containing
            atom coordinates for tetrahedral configurations.
        
        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size, n_loss, 6) containing
            the 6 pairwise distances of each tetrahedron.
        """
        idx_pairs = torch.tensor([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], device=positions.device)
        
        a = positions[:, :, idx_pairs[:, 0], :]
        b = positions[:, :, idx_pairs[:, 1], :]
        return torch.norm(a - b, dim=-1)
    
    def _eval_loss(self, positions: torch.tensor) -> torch.tensor:
        """
        Evaluate the loss for a batch of atom positions.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in Angstroms.
        
        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size,) containing losses.
        """
        all_losses = []
        
        for group in self._kde_groups:
            idxs = group["atom_indices"]           # [n_loss, 4]
            weights = group["weights"]             # [n_loss]
            offsets = group["offsets"]             # [n_loss]
            kde = group["kde"]
            
            batch_size = positions.shape[0]
            n_loss = idxs.shape[0]
            
            batch_idx = torch.arange(batch_size, device=self.device).view(-1, 1, 1).expand(batch_size, n_loss, 4)
            atom_idx = idxs.unsqueeze(0).expand(batch_size, -1, 4)
            
            tetra_pos = positions[batch_idx, atom_idx]  # [batch, n_loss, 4, 3]
            dists = self._get_batch_geometry(tetra_pos)  # [batch, n_loss, 6]
            
            # KDE score: flatten to 2D → evaluate → reshape back
            logP_flat = kde.score_samples(dists.view(-1, 6))  # [batch * n_loss]
            logP = logP_flat.view(batch_size, n_loss)
            
            energy = -KB * self.temp * logP
            scaled = energy * weights + offsets  # broadcast [n_loss] over [batch, n_loss]
            all_losses.append(scaled)
        
        if not all_losses:
            # Handle case where no valid cyclizations were found
            return torch.zeros(positions.shape[0], device=self.device)
            
        all_losses = torch.cat(all_losses, dim=1)  # [batch, total_losses]
        
        # === Group reduction using scatter ===
        group_idx_tensor = self._group_idx_tensor  # [n_total_losses]
        n_groups = self._n_groups
        batch_size = all_losses.shape[0]
        
        batch_idx = torch.arange(batch_size, device=self.device).view(-1, 1).expand(batch_size, group_idx_tensor.size(0))
        group_idx_expand = group_idx_tensor.view(1, -1).expand(batch_size, -1)
        
        flat_idx = batch_idx * n_groups + group_idx_expand  # [batch, total_losses]
        flat_losses = all_losses.view(-1)
        flat_out = torch.full((batch_size * n_groups,), float('inf'), device=self.device)
        
        flat_out = flat_out.scatter_reduce(0, flat_idx.reshape(-1), flat_losses, reduce='amin')
        group_min = flat_out.view(batch_size, n_groups)  # [batch, n_groups]
        
        return soft_min(group_min, alpha=self.alpha)
    
    def __call__(self, positions: torch.tensor) -> torch.tensor:
        """
        Evaluate the loss for a batch of atom positions.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in input units.
        
        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size,) containing losses.
        
        Examples
        --------
        >>> handler = ChemicalLossHandler.from_pdb("peptide.pdb")
        >>> positions = torch.randn(10, 100, 3)  # 10 structures, 100 atoms, 3D
        >>> loss = handler(positions)
        >>> print(loss.shape)
        torch.Size([10])
        """
        return super().__call__(positions) # automatically handles unit conversion.
    
    def get_smallest_loss(self, positions: torch.tensor) -> List[ChemicalLoss]:
        """
        Get the loss with the smallest value for each structure in the batch.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in input units.
        
        Returns
        -------
        List[ChemicalLoss]
            List of loss objects with the smallest value for each structure.
        
        Examples
        --------
        >>> handler = ChemicalLossHandler.from_pdb("peptide.pdb")
        >>> positions = torch.randn(10, 100, 3)
        >>> best_losses = handler.get_smallest_loss(positions)
        >>> print(best_losses[0].method)  # Print method for first structure
        'Disulfide, CYS 3 -> CYS 14'
        """
        positions_ang = self._convert_positions(positions)
        batched_losses = torch.stack([loss(positions_ang) for loss in self.losses], dim=1)
        
        if batched_losses.dim() == 1:  # Single structure
            batched_losses = batched_losses.unsqueeze(0)
            
        min_indices = torch.argmin(batched_losses, dim=1)
        return [self.losses[i] for i in min_indices.tolist()]
    
    def get_smallest_loss_methods(self, positions: torch.tensor) -> List[str]:
        """
        Get the method names of the smallest losses for each structure.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in input units.
        
        Returns
        -------
        List[str]
            List of method strings for the smallest losses.
        """
        return [loss.method for loss in self.get_smallest_loss(positions)]
    
    def get_all_losses(self, positions: torch.tensor) -> torch.tensor:
        """
        Get all individual loss values for each structure in the batch.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in input units.
        
        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size, n_losses) containing all loss values.
        """
        positions_ang = self._convert_positions(positions)
        return torch.stack([loss(positions_ang) for loss in self.losses], dim=1)
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of all detected cyclization options,
        including information about resonance groups.
        
        Returns
        -------
        str
            Formatted string with cyclization information and resonance groups.
        
        Examples
        --------
        >>> handler = ChemicalLossHandler.from_pdb("peptide.pdb")
        >>> print(handler.summary())
        ChemicalLossHandler Summary
        ==========================
        PDB File: peptide.pdb
        Total cyclization options: 12
        Total resonance groups: 8
        
        Cyclization types:
        - Disulfide: 3 options
        * CYS 3 -> CYS 14
        * CYS 3 -> CYS 38
        * CYS 14 -> CYS 38
        
        - AspGlu: 2 options (1 resonance group)
        * ASP 7 -> GLU 21 (OE1) [resonance group 1]
        * ASP 7 -> GLU 21 (OE2) [resonance group 1]
        
        [...]
        
        Resonance Groups:
        1. ASP 7 -> GLU 21 (2 variants)
        2. ASP 29 -> GLU 33 (2 variants)
        [...]
        """
        strategy_counts = {}
        strategy_details = {}
        
        # Create a mapping from resonance group ID to the losses in that group
        resonance_groups = {}
        for i, loss in enumerate(self.losses):
            group_id = self._resonance_group_ids[i].item()
            if group_id not in resonance_groups:
                resonance_groups[group_id] = []
            resonance_groups[group_id].append(loss)
        
        # Count the number of actual resonance groups (groups with more than 1 loss)
        actual_resonance_groups = {k: v for k, v in resonance_groups.items() if len(v) > 1}
        
        # Organize losses by strategy
        for loss in self.losses:
            method = loss.method
            strategy = method.split(",")[0].strip()
            
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
                strategy_details[strategy] = []
                
            strategy_counts[strategy] += 1
            
            # Find which resonance group this loss belongs to
            group_idx = None
            for i, loss_obj in enumerate(self.losses):
                if loss_obj == loss:
                    group_idx = self._resonance_group_ids[i].item()
                    break
            
            # Check if this is part of a resonance group with multiple losses
            if group_idx in actual_resonance_groups and len(actual_resonance_groups[group_idx]) > 1:
                # Extract residue info from the method
                detail = method.split(",", 1)[1].strip()
                # Add group info to the detail
                resonance_count = len(actual_resonance_groups[group_idx])
                resonance_idx = list(actual_resonance_groups.keys()).index(group_idx) + 1
                detail = f"{detail} [resonance group {resonance_idx}]"
            else:
                detail = method.split(",", 1)[1].strip()
                
            strategy_details[strategy].append(detail)
        
        # Build the summary string
        summary = [
            f"ChemicalLossHandler Summary",
            f"==========================",
            f"PDB File: {self.pdb_path.name}",
            f"Total cyclization options: {len(self.losses)}",
            f"Total resonance groups: {len(actual_resonance_groups)}",
            f"",
            f"Cyclization types:"
        ]
        
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            # Check if this strategy has any resonance groups
            has_resonance = False
            for detail in strategy_details[strategy]:
                if "[resonance group" in detail:
                    has_resonance = True
                    break
                    
            if has_resonance:
                resonance_info = f" (with resonance variants)"
            else:
                resonance_info = ""
                
            summary.append(f"- {strategy}: {count} options{resonance_info}")
            
            for i, detail in enumerate(strategy_details[strategy][:5]):
                summary.append(f"  * {detail}")
                    
            if len(strategy_details[strategy]) > 5:
                remaining = len(strategy_details[strategy]) - 5
                summary.append(f"  * ... and {remaining} more")
                    
            summary.append("")
        
        # Add a section for resonance groups if there are any
        if actual_resonance_groups:
            summary.append("Resonance Groups:")
            
            for idx, (group_id, losses) in enumerate(actual_resonance_groups.items()):
                # Extract a representative method/description for this resonance group
                base_method = losses[0].method.split(",", 1)[1].strip()
                # Remove any specific atom names (e.g., "(OE1)") that might differ between resonance structures
                base_method = re.sub(r'\([A-Z0-9]+\)', '', base_method).strip()
                
                # List the specific atom variations in this resonance group
                atom_variants = []
                for loss in losses:
                    # Try to extract the specific atom name from the method string
                    atom_match = re.search(r'\(([A-Z0-9]+)\)', loss.method)
                    atom_name = atom_match.group(1) if atom_match else "?"
                    atom_variants.append(atom_name)
                
                variant_str = ", ".join(atom_variants) if atom_variants else "multiple variants"
                summary.append(f"{idx+1}. {base_method} ({len(losses)} variants: {variant_str})")
            
            summary.append("")
        
        return "\n".join(summary)
    
    ### debugging methods -- I will keep these in because they will be useful for user
    ### diagnostics as well

    def validate_configuration(self) -> bool:
        """
        Validate the handler configuration and provide verbose diagnostics.
        
        Returns
        -------
        bool
            True if configuration is valid, False otherwise.
        """
        valid = True
        messages = ["Validating ChemicalLossHandler configuration:"]
        
        # Check for strategies
        if not self.strategies:
            valid = False
            messages.append("✗ ERROR: No cyclization strategies provided")
        else:
            messages.append(f"✓ Found {len(self.strategies)} cyclization strategies")
        
        # Check for losses
        if not self.losses:
            valid = False
            messages.append("✗ ERROR: No valid cyclization sites found in structure")
        else:
            messages.append(f"✓ Found {len(self.losses)} potential cyclization sites")
        
        # Check KDE groups
        if not hasattr(self, '_kde_groups') or not self._kde_groups:
            valid = False
            messages.append("✗ ERROR: No KDE groups initialized")
        else:
            messages.append(f"✓ Found {len(self._kde_groups)} KDE groups")
        
        # Test small calculation to verify computation pipeline
        if valid:
            try:
                # Create a small test batch (2 structures with same atoms as loaded PDB)
                n_atoms = self.traj.xyz.shape[1]
                test_batch = torch.rand(2, n_atoms, 3, device=self.device)
                
                # Run a forward pass
                result = self(test_batch)
                
                if result.shape != (2,):
                    valid = False
                    messages.append(f"✗ ERROR: Expected output shape (2,), got {result.shape}")
                else:
                    messages.append("✓ Successfully ran test calculation")
                    
            except Exception as e:
                valid = False
                messages.append(f"✗ ERROR: Test calculation failed with: {str(e)}")
        
        # Print diagnostic messages
        print("\n".join(messages))
        if not valid:
            print("\nChemicalLossHandler validation FAILED. See errors above.")
        else:
            print("\nChemicalLossHandler validation PASSED.")
        
        return valid
    
    def inspect_losses(self, positions: torch.tensor, top_k: int = 5) -> None:
        """
        Inspect the top contributing loss terms for debugging.
        
        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) in input units.
        
        top_k : int, optional
            Number of top loss terms to display. Default is 5.
        """
        if not self.losses:
            print("No losses available to inspect. Check if strategies were provided.")
            return
        
        positions_ang = self._convert_positions(positions)
        all_losses = torch.stack([loss(positions_ang) for loss in self.losses], dim=1)
        
        batch_size = positions.shape[0]
        
        print(f"Loss inspection for {batch_size} structures:")
        print("-" * 50)
        
        for i in range(min(batch_size, 3)):  # Show at most 3 structures
            print(f"\nStructure {i}:")
            
            # Get the overall combined loss
            combined_loss = self(positions[i:i+1]).item()
            print(f"Combined soft-min loss: {combined_loss:.4f}")
            
            # Get top contributing individual losses
            struct_losses = all_losses[i]
            top_values, top_indices = torch.topk(struct_losses, min(top_k, len(self.losses)), largest=False)
            
            print("\nTop contributing loss terms:")
            for j, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
                loss_obj = self.losses[idx]
                print(f"{j+1}. {loss_obj.method}: {val:.4f}")
        
        print("\nNote: Lower values indicate more favorable cyclization configurations.")
    
    def visualize_cyclization_sites(self, output_path: Optional[str] = None): # TODO: clean this up.
        """
        Generate a visualization of cyclization sites.
        
        This method creates a PyMOL or NGLView visualization showing the
        potential cyclization sites in the structure.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the visualization. If None, displays interactive view.
        
        Returns
        -------
        viewer : object
            Visualization object (NGLView or PyMOL script)
        
        Notes
        -----
        Requires nglview or pymol packages to be installed.
        """
        try:
            import nglview as nv
            
            # Create a view from the trajectory
            view = nv.show_mdtraj(self.traj)
            
            # Add representations for cyclization sites
            for loss in self.losses:
                # Extract atom indices
                atom_idxs = [loss.atom_idxs[k] for k in loss.atom_idxs_keys]
                
                # Add representation for this cyclization option
                view.add_representation('ball+stick', selection=atom_idxs, color='skyblue')
                
            if output_path:
                view.render_image(output_path)
            
            return view
            
        except ImportError:
            print("Visualization requires nglview package. Please install with: pip install nglview")
            return None
        
    @classmethod
    def check_dependencies(cls) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns
        -------
        bool
            True if all dependencies are available, False otherwise.
        """
        missing = []
        
        # Check for required packages
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        try:
            import mdtraj
        except ImportError:
            missing.append("mdtraj")
        
        # Add other dependencies as needed
        
        if missing:
            print(f"Missing required dependencies: {', '.join(missing)}")
            print("Please install them with pip:")
            print(f"pip install {' '.join(missing)}")
            return False
        
        return True