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
    
    Resonant structures (multiple atom choices for the same chemical interaction)
    are handled by taking the minimum loss within each resonance group before
    applying the soft minimum across all unique cyclization opportunities.

    Parameters
    ----------
    pdb_path : Union[str, Path]
        Path to the PDB file containing the peptide structure.
    units_factor : float
        Conversion factor to convert input coordinates to Angstroms.
    strategies : Optional[Set[Type[ChemicalLoss]]]
        Set of chemical loss strategies to consider. If None, uses default_strategies.
    weights : Optional[Dict[Type[ChemicalLoss], float]]
        Dictionary mapping loss types to their weights. If None, all weights are 1.0.
    offsets : Optional[Dict[Type[ChemicalLoss], float]]
        Dictionary mapping loss types to their offsets. If None, all offsets are 0.0.
    temp : float, default=300.0
        Temperature in Kelvin for energy calculations.
    alpha : float, default=-3.0
        Soft minimum parameter controlling the sharpness of the minimum.
    mask : Optional[Set[int]]
        Set of residue indices to exclude from cyclization consideration.
    device : Optional[torch.device]
        Device to use for computations. If None, uses CUDA if available, else CPU.

    Attributes
    ----------
    default_strategies : Set[Type[ChemicalLoss]]
        Default set of cyclization chemistries to consider.
    bonding_atoms : Set[str]
        Common atom names involved in cyclization reactions.
    units_factors_dict : Dict[str, float]
        Dictionary of unit conversion factors.
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
                mask: Optional[Set[int]] = None,
                device: Optional[torch.device] = None,
                **kwargs) -> "ChemicalLossHandler":
        """
        Create a ChemicalLossHandler from a PDB file with simplified parameters.
        
        This is the recommended way to initialize a ChemicalLossHandler as it handles
        unit conversion automatically.

        Parameters
        ----------
        pdb_path : Union[str, Path]
            Path to the PDB file containing the peptide structure.
        units : Optional[str]
            Name of the input coordinate units (e.g., 'nm', 'angstrom'). Must be one of
            the keys in units_factors_dict.
        units_factor : Optional[float]
            Direct conversion factor to convert input coordinates to Angstroms.
            Either units or units_factor must be provided, but not both.
        temp : float, default=300.0
            Temperature in Kelvin for energy calculations.
        alpha : float, default=-3.0
            Soft minimum parameter controlling the sharpness of the minimum.
        mask : Optional[Set[int]]
            Set of residue indices to exclude from cyclization consideration.
        device : Optional[torch.device]
            Device to use for computations. If None, uses CUDA if available, else CPU.
        **kwargs : dict
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        ChemicalLossHandler
            A new instance of ChemicalLossHandler.

        Raises
        ------
        ValueError
            If both units and units_factor are provided, or if neither is provided.
            If units is not a valid unit name.
        FileNotFoundError
            If the PDB file does not exist.
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
            mask=mask,
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
                 mask: Optional[Set[int]] = None, 
                 # can mask certain residues from consideration when initializing losses.
                 device: Optional[torch.device] = None,
                 ):
        """
        Initialize a ChemicalLossHandler with detailed control over parameters.
        
        For most use cases, the `from_pdb` class method provides a simpler interface.
        """
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
        
        self._mask: Set[int] = mask or set()
        self._validate_mask()

        # Initialize losses and build resonance groups
        self._initialize_losses()
        
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
    
    @staticmethod
    def precompute_atom_indices(residues, atom_names) -> Dict[Tuple[int, str], int]:
        """
        Precompute a dictionary mapping (residue_index, atom_name) to atom indices.
        """
        indices = {}
        for residue in residues:
            for name in atom_names:
                atom = next((a for a in residue.atoms if a.name == name), None)
                if atom:
                    indices[(residue.index, name)] = atom.index
        return indices
    
    def _validate_mask(self) -> None:
        """
        Validate the mask indices based on the topology and notify about masked residues.
        
        This method checks if all masked residue indices are valid and prints
        information about which residues are being masked from cyclization chemistry.

        Raises
        ------
        ValueError
            If any residue index in the mask is invalid (not present in the topology).
        """
        # Validate mask indices if mask is not empty
        if self._mask:
            valid_indices = {res.index for res in self._topology.residues}
            invalid_indices = self._mask - valid_indices
            if invalid_indices:
                raise ValueError(
                    f"Invalid residue indices in mask: {invalid_indices}. "
                    f"Valid indices range from 0 to {len(valid_indices)-1}."
                )
            
            # Get residue information for masked residues
            masked_residues = []
            for res in self._topology.residues:
                if res.index in self._mask:
                    masked_residues.append(f"{res.name} {res.index}")
            
            if masked_residues:
                print(f"\nMasking {len(masked_residues)} residues from cyclization chemistry:")
                for res in masked_residues:
                    print(f"  - {res}")
                print("These residues will not participate in any cyclization reactions.\n")
        else:
            pass
    
    def _initialize_losses(self) -> None:
        """
        Initialize all loss functions and organize them into resonance groups.
        
        This method:
        1. Creates loss instances for each strategy and valid residue pair
        2. Groups losses by resonance (same method between same residues but different atoms)
        3. Excludes any losses involving masked residues
        4. Stores both the grouped structure and a flat list of all losses

        Raises
        ------
        ValueError
            If no cyclization strategies are provided and default_strategies is empty.
        """
        if not self.strategies:
            raise ValueError(
                "No cyclization strategies provided and default_strategies is empty. "
                "The ChemicalLossHandler requires at least one strategy to function."
            )

        # Get the loaded trajectory
        traj = self.traj
        
        # Pre-compute atom indices for faster lookup
        atom_idx_dict = self.precompute_atom_indices(
            list(self.topology.residues),
            self.bonding_atoms
        )
        
        # Dictionary to group losses by resonance key
        # Key: (method_base, frozenset(residue_indices))
        # Value: List of ChemicalLoss instances
        resonance_groups: Dict[Tuple[str, frozenset], List[ChemicalLoss]] = {}
        
        # Create all loss instances
        for strat in self.strategies:
            for idxs_method_pair in strat.get_indexes_and_methods(traj, atom_idx_dict):
                # Skip if either residue in the pair is masked
                if self._mask and (idxs_method_pair.pair & self._mask):
                    continue
                
                # Extract the base method (without resonance info) for grouping
                method_base = idxs_method_pair.method.split(" (")[0]  # Remove resonance info
                resonance_key = (method_base, frozenset(idxs_method_pair.pair))
                
                # Create loss instance
                loss = strat(
                    method=idxs_method_pair.method,
                    atom_idxs=idxs_method_pair.indexes,
                    temp=self.temp,
                    weight=self.weights[strat],
                    offset=self.offsets[strat],
                    resonance_key=resonance_key,
                    device=self.device,
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
    
    def _eval_loss(self, positions: torch.tensor) -> torch.tensor:
        """
        Evaluate the loss for a batch of atom positions.
        
        For each resonance group, computes the minimum loss among all resonant
        structures, then applies soft minimum across all unique cyclization opportunities.

        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions in Angstroms.

        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size,) containing the combined loss values.
            Returns zeros if no valid cyclizations were found.
        """
        if not self._resonance_groups:
            # Handle case where no valid cyclizations were found
            return torch.zeros(positions.shape[0], device=self.device)
        
        batch_size = positions.shape[0]
        group_losses = []
        
        # Process each resonance group
        for resonance_key, losses_in_group in self._resonance_groups.items():
            # Compute losses for all instances in this resonance group
            group_loss_values = []
            
            for loss in losses_in_group:
                # Extract tetrahedral vertex positions
                vertex_indices = [loss.atom_idxs[key] for key in loss.atom_idxs_keys]
                
                # Get positions for all 4 vertices: shape [batch_size, 4, 3]
                vertex_positions = positions[:, vertex_indices, :]
                
                # Compute pairwise distances for tetrahedral geometry
                # Using the same distance calculation as in ChemicalLoss._eval_loss
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                # Evaluate KDE and convert to energy
                logP = loss.kde_pdf.score_samples(dists)
                energy = -KB * self.temp * logP
                
                # Apply weight and offset
                scaled_energy = energy * loss.weight + loss.offset
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
            return group_losses[0]
        else:
            all_group_losses = torch.stack(group_losses, dim=1)  # [batch_size, n_groups]
            return soft_min(all_group_losses, alpha=self.alpha)
    
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
        """
        return super().__call__(positions)  # automatically handles unit conversion
    
    def get_smallest_loss(self, positions: torch.tensor) -> List[ChemicalLoss]:
        """
        Get the loss with the smallest value for each structure in the batch.
        
        This method considers resonance grouping, returning the best loss
        (minimum value) for each structure.

        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions.

        Returns
        -------
        List[ChemicalLoss]
            List of ChemicalLoss instances, one for each structure in the batch,
            representing the most favorable cyclization option.
        """
        positions_ang = self._convert_positions(positions)
        
        batch_size = positions_ang.shape[0]
        best_losses = []
        
        for batch_idx in range(batch_size):
            single_pos = positions_ang[batch_idx:batch_idx+1]  # Keep batch dimension
            
            best_loss_value = float('inf')
            best_loss = None
            
            # Check each resonance group
            for resonance_key, losses_in_group in self._resonance_groups.items():
                group_loss_values = []
                
                # Evaluate all losses in this group
                for loss in losses_in_group:
                    loss_value = loss(single_pos).item()
                    group_loss_values.append((loss_value, loss))
                
                # Find the best (minimum) loss in this group
                group_best_value, group_best_loss = min(group_loss_values, key=lambda x: x[0])
                
                # Check if this is the overall best
                if group_best_value < best_loss_value:
                    best_loss_value = group_best_value
                    best_loss = group_best_loss
            
            best_losses.append(best_loss)
        
        return best_losses
    
    def get_smallest_loss_methods(self, positions: torch.tensor) -> List[str]:
        """
        Get the method names of the smallest losses for each structure.

        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions.

        Returns
        -------
        List[str]
            List of method names, one for each structure in the batch,
            describing the most favorable cyclization option.
        """
        return [loss.method for loss in self.get_smallest_loss(positions)]
    
    def get_all_losses(self, positions: torch.tensor) -> torch.tensor:
        """
        Get all individual loss values for each structure in the batch.
        
        This returns the minimum loss for each resonance group, considering
        all possible cyclization options.

        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions.

        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size, n_groups) containing the minimum loss
            for each resonance group.
        """
        positions_ang = self._convert_positions(positions)
        batch_size = positions_ang.shape[0]
        
        group_losses = []
        
        for resonance_key, losses_in_group in self._resonance_groups.items():
            group_loss_values = []
            
            for loss in losses_in_group:
                loss_values = loss(positions_ang)  # [batch_size]
                group_loss_values.append(loss_values)
            
            if len(group_loss_values) == 1:
                group_min = group_loss_values[0]
            else:
                group_stack = torch.stack(group_loss_values, dim=1)  # [batch_size, n_resonant]
                group_min = torch.min(group_stack, dim=1)[0]  # [batch_size]
            
            group_losses.append(group_min)
        
        return torch.stack(group_losses, dim=1)  # [batch_size, n_groups]
    
    def summary(self) -> str:
        """
        Generate a comprehensive human-readable summary of all detected cyclization options.
        
        The summary includes:
        - Overview of total cyclization options and groups
        - KDE models used and their statistics
        - Strategy configuration details
        - Detailed analysis of each cyclization group
        - Statistics about resonant vs non-resonant groups

        Returns
        -------
        str
            A formatted string containing the complete summary.
        """
        if not self._resonance_groups:
            return "No valid cyclization sites found."
        
        total_losses = len(self._losses)
        total_groups = len(self._resonance_groups)
        
        # Collect KDE information
        kde_info = {}
        for loss in self._losses:
            kde_file = loss.kde_file
            if kde_file not in kde_info:
                kde_info[kde_file] = {
                    'count': 0,
                    'strategies': set(),
                    'losses': []
                }
            kde_info[kde_file]['count'] += 1
            kde_info[kde_file]['strategies'].add(type(loss).__name__)
            kde_info[kde_file]['losses'].append(loss)
        
        summary = [
            f"ChemicalLossHandler Summary",
            f"=" * 50,
            f"PDB File: {self.pdb_path.name}",
            f"Temperature: {self.temp}K",
            f"Soft minimum alpha: {self.alpha}",
            f"Device: {self.device}",
            f"",
            f"Overview:",
            f"  Total cyclization options: {total_losses}",
            f"  Unique cyclization groups (after resonance): {total_groups}",
            f"  KDE models used: {len(kde_info)}",
            f"  Strategies employed: {len(self.strategies)}",
            f""
        ]
        
        # KDE Models Section
        summary.extend([
            f"KDE Models:",
            f"-" * 20
        ])
        
        for i, (kde_file, info) in enumerate(kde_info.items(), 1):
            kde_path = Path(kde_file)
            summary.extend([
                f"{i}. {kde_path.name}",
                f"   Full path: {kde_file}",
                f"   Used by {info['count']} loss instances",
                f"   Strategies: {', '.join(sorted(info['strategies']))}",
                f""
            ])
        
        # Strategies Section
        summary.extend([
            f"Strategy Configuration:",
            f"-" * 25
        ])
        
        for strategy in sorted(self.strategies, key=lambda x: x.__name__):
            weight = self.weights.get(strategy, 1.0)
            offset = self.offsets.get(strategy, 0.0)
            count = sum(1 for loss in self._losses if isinstance(loss, strategy))
            
            summary.extend([
                f"• {strategy.__name__}:",
                f"  Weight: {weight}, Offset: {offset}",
                f"  Instances: {count}",
                f""
            ])
        
        # Detailed Resonance Groups Section
        summary.extend([
            f"Detailed Cyclization Analysis:",
            f"-" * 35
        ])
        
        # Group by strategy for organization
        strategy_groups = {}
        for (method_base, residue_pair), losses_in_group in self._resonance_groups.items():
            strategy = method_base.split(",")[0].strip()
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(((method_base, residue_pair), losses_in_group))
        
        for strategy_name in sorted(strategy_groups.keys()):
            groups = strategy_groups[strategy_name]
            summary.extend([
                f"{strategy_name} ({len(groups)} unique sites):",
                f"{'=' * (len(strategy_name) + 20)}"
            ])
            
            for group_idx, ((method_base, residue_pair), losses_in_group) in enumerate(groups, 1):
                is_resonant = len(losses_in_group) > 1
                
                summary.append(f"{group_idx}. {method_base}")
                summary.append(f"   Residues involved: {sorted(residue_pair)}")
                
                if is_resonant:
                    summary.append(f"   Resonance group with {len(losses_in_group)} variants:")
                    
                    for variant_idx, loss in enumerate(losses_in_group, 1):
                        # Extract atom indices for this variant
                        atom_info = []
                        for key in loss.atom_idxs_keys:
                            atom_idx = loss.atom_idxs[key]
                            atom_info.append(f"{key}:{atom_idx}")
                        
                        # Extract specific resonance info from method string
                        resonance_detail = ""
                        if " (" in loss.method and ")" in loss.method:
                            resonance_detail = loss.method.split(" (")[1].rstrip(")")
                            resonance_detail = f" ({resonance_detail})"
                        
                        summary.extend([
                            f"     {chr(96 + variant_idx)}. {loss.method}",
                            f"        Atom indices: {', '.join(atom_info)}",
                            f"        Weight: {loss.weight}, Offset: {loss.offset}",
                            f"        KDE model: {Path(loss.kde_file).name}"
                        ])
                else:
                    # Single variant
                    loss = losses_in_group[0]
                    atom_info = []
                    for key in loss.atom_idxs_keys:
                        atom_idx = loss.atom_idxs[key]
                        atom_info.append(f"{key}:{atom_idx}")
                    
                    summary.extend([
                        f"   Single variant:",
                        f"     Atom indices: {', '.join(atom_info)}",
                        f"     Weight: {loss.weight}, Offset: {loss.offset}",
                        f"     KDE model: {Path(loss.kde_file).name}"
                    ])
                
                summary.append("")  # Blank line between groups
        
        # Statistics Section
        resonant_groups = sum(1 for losses in self._resonance_groups.values() if len(losses) > 1)
        non_resonant_groups = total_groups - resonant_groups
        
        summary.extend([
            f"Statistics:",
            f"-" * 15,
            f"Resonant groups (multiple variants): {resonant_groups}",
            f"Non-resonant groups (single variant): {non_resonant_groups}",
            f"Average variants per resonant group: {total_losses / max(resonant_groups, 1):.1f}" if resonant_groups > 0 else "Average variants per resonant group: N/A",
            f"",
            f"Unit conversion factor: {self.units_factor} (input units → Angstroms)",
            f""
        ])
        
        return "\n".join(summary)
    
    # Keep existing debugging and utility methods unchanged
    def validate_configuration(self) -> bool:
        """
        Validate the handler configuration and provide verbose diagnostics.
        
        Performs several checks:
        1. Verifies that strategies are provided
        2. Confirms that valid cyclization sites were found
        3. Tests a sample calculation
        4. Validates output shapes

        Returns
        -------
        bool
            True if all validation checks pass, False otherwise.
        """
        valid = True
        messages = ["Validating ChemicalLossHandler configuration:"]
        
        if not self.strategies:
            valid = False
            messages.append("✗ ERROR: No cyclization strategies provided")
        else:
            messages.append(f"✓ Found {len(self.strategies)} cyclization strategies")
        
        if not self._resonance_groups:
            valid = False
            messages.append("✗ ERROR: No valid cyclization sites found in structure")
        else:
            messages.append(f"✓ Found {len(self._resonance_groups)} unique cyclization groups")
            messages.append(f"✓ Found {len(self._losses)} total cyclization options")
        
        if valid:
            try:
                n_atoms = self.traj.xyz.shape[1]
                test_batch = torch.rand(2, n_atoms, 3, device=self.device)
                result = self(test_batch)
                
                if result.shape != (2,):
                    valid = False
                    messages.append(f"✗ ERROR: Expected output shape (2,), got {result.shape}")
                else:
                    messages.append("✓ Successfully ran test calculation")
                    
            except Exception as e:
                valid = False
                messages.append(f"✗ ERROR: Test calculation failed with: {str(e)}")
        
        print("\n".join(messages))
        if not valid:
            print("\nChemicalLossHandler validation FAILED. See errors above.")
        else:
            print("\nChemicalLossHandler validation PASSED.")
        
        return valid
    
    def inspect_losses(self, positions: torch.tensor, top_k: int = 5) -> None:
        """
        Inspect the top contributing loss terms for debugging purposes.
        
        For each structure in the batch (up to 3), prints:
        - The combined soft-min loss
        - The top K contributing cyclization groups
        - Detailed information about each group

        Parameters
        ----------
        positions : torch.tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions.
        top_k : int, default=5
            Number of top contributing losses to display for each structure.
        """
        if not self._resonance_groups:
            print("No resonance groups available to inspect.")
            return
        
        positions_ang = self._convert_positions(positions)
        batch_size = positions.shape[0]
        
        print(f"Loss inspection for {batch_size} structures:")
        print("-" * 50)
        
        for i in range(min(batch_size, 3)):  # Show at most 3 structures
            print(f"\nStructure {i}:")
            
            combined_loss = self(positions[i:i+1]).item()
            print(f"Combined soft-min loss: {combined_loss:.4f}")
            
            # Get losses from each resonance group
            group_results = []
            for (method_base, residue_pair), losses_in_group in self._resonance_groups.items():
                group_values = []
                group_losses_obj = []
                
                for loss in losses_in_group:
                    single_pos = positions_ang[i:i+1]
                    loss_value = loss(single_pos).item()
                    group_values.append(loss_value)
                    group_losses_obj.append(loss)
                
                min_idx = group_values.index(min(group_values))
                best_loss_obj = group_losses_obj[min_idx]
                best_value = group_values[min_idx]
                
                group_results.append((best_value, best_loss_obj))
            
            # Sort and show top contributors
            group_results.sort(key=lambda x: x[0])
            
            print("\nTop contributing cyclization groups:")
            for j, (val, loss_obj) in enumerate(group_results[:top_k]):
                print(f"{j+1}. {loss_obj.method}: {val:.4f}")
        
        print("\nNote: Lower values indicate more favorable cyclization configurations.")