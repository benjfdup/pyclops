"""
This module contains the ChemicalLoss class, which is the base class for all chemical losses.
ChemicalLoss instances are immutable after initialization.
ChemicalLoss instances should not generally be called directly. Use ChemicalLossHandler instead.
ChemicalLoss instances are used to compute the loss for a given set of atom positions.
"""
__all__ = ['ChemicalLoss']

from abc import ABC
from typing import Dict, Optional, Tuple, ClassVar, final, Sequence, List, Any
from itertools import product
import warnings

import torch
import mdtraj as md

from ...utils.constants import KB
from ...torchkde import KernelDensity
from .kde_manager import KDEManager

# type aliases
AtomIndexKeys = Tuple[str, str, str, str]

AtomIndexMap = Dict[str, int]  # maps atom names to their global indices
ResonanceKey = Tuple[str, frozenset[int]] # loss_method (str), set of ints of size 2 (Amino Acid idxs)

AtomKey = Tuple[int, str]  # (residue_idx, atom_name)
AtomIndexDict = Dict[AtomKey, int]

class ChemicalLoss(ABC):
    """
    ChemicalLoss is largely for use in the ChemicalLossHandler class, which handles position 
    conversions to Angstroms. Subclasses must override cls.get_loss_instances, 
    and must provide cls._method, cls._kde_file, and cls._atom_idxs_keys.

    Instances of this class should be treated as immutable after initialization.
    All provided positions should be in Angstroms. This is not checked by the class.
    """
    
    # Class variables to be overridden by subclasses
    _atom_idxs_keys: ClassVar[AtomIndexKeys] = ()
    _kde_file: ClassVar[str] = ''
    _method: ClassVar[str] = ''

    # initialization methods
    def __init__(
            self,
            atom_idxs: AtomIndexMap,
            temp: float = 1.0,
            weight: float = 1.0,
            offset: float = 0.0,
            resonance_key: Optional[ResonanceKey] = None, # used to determine what losses are resonant with each other
            # the string will represent the type of the chemical loss. The frozenset represents the amino acid idxs which
            # loss is between (should be of len 2)
            device: Optional[torch.device] = None
    ):
        # validate variables
        self.__validate_class_variables()
        self.__validate_init_inputs(atom_idxs, temp, weight, offset, resonance_key, device)

        # Initialize instance variables and resources
        self._initialize_resources(atom_idxs, temp, weight, offset, resonance_key, device)

    def __validate_class_variables(self) -> None:
        if len(self._atom_idxs_keys) != 4:
            raise ValueError(f"ChemicalLoss requires exactly 4 atom keys, got {len(self._atom_idxs_keys)}.")
        if not self._kde_file:
            raise ValueError(f"Subclass {self.__class__.__name__} must specify kde_file attribute")
        if self._method == '':
            raise ValueError(f"Subclass {self.__class__.__name__} must specify method attribute")

    def __validate_init_inputs(self, # double underscore because I almost accidentally overrode the parent class's method in a subclass. 
                                # oops.
                         atom_idxs: AtomIndexMap,
                         temp: float,
                         weight: float,
                         offset: float,
                         resonance_key: Optional[ResonanceKey],
                         device: Optional[torch.device],
                         ) -> None:
        
        # check that the atom keys are correct
        if set(atom_idxs.keys()) != set(self._atom_idxs_keys):
            raise ValueError(f"Expected atom keys {self._atom_idxs_keys}, got {set(atom_idxs.keys())}")
        # check that the atom keys are integers
        if not all(isinstance(val, int) for val in atom_idxs.values()):
            raise ValueError("All atom_idxs values must be integers")
        # check that the atom values are unique
        if len(set(atom_idxs.values())) != len(atom_idxs.values()):
            raise ValueError("All atom_idxs values must be unique")
        if len(atom_idxs.keys()) != len(self._atom_idxs_keys):
            raise ValueError(f"Expected 4 atom keys, got {len(atom_idxs.keys())}")
        # check that the temperature is a positive float
        if temp <= 0.0:
            raise ValueError("Temperature must be a positive float")
        # check that the weight is a positive float
        if weight <= 0.0:
            raise ValueError("Weight must be a positive float")
        # check that the offset is a float or is an int
        if not isinstance(offset, (float, int)):
            raise ValueError("Offset must be a float")
        # check that the resonance key is valid
        if resonance_key is not None:
            if not isinstance(resonance_key, tuple) or len(resonance_key) != 2:
                raise ValueError("Resonance key must be a tuple of length 2")
            if not isinstance(resonance_key[0], str):
                raise ValueError(f"resonance_key[0] must be a string. Got {type(resonance_key[0])}")
            if not isinstance(resonance_key[1], frozenset):
                raise ValueError(f"resonance_key[1] must be a frozenset. Got {type(resonance_key[1])}")
            if len(resonance_key[1]) != 2:
                raise ValueError(f"resonance_key[1] must be a set of size 2. Got {len(resonance_key[1])}")
            if not all(isinstance(val, int) for val in resonance_key[1]):
                raise ValueError("All resonance key values must be integers")
        # validate device
        if device is not None:
            if not isinstance(device, torch.device):
                raise ValueError(f"if provided, device must be a torch.device. Got {type(device)}")
            
    def _initialize_resources(self,
                             atom_idxs: AtomIndexMap,
                             temp: float,
                             weight: float,
                             offset: float,
                             resonance_key: Optional[ResonanceKey],
                             device: Optional[torch.device]) -> None:
        """Initialize instance variables and load required resources."""
        # store variables
        self._atom_idxs = atom_idxs
        self._temp = float(temp)
        self._weight = float(weight)
        self._offset = float(offset)
        self._resonance_key = resonance_key
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Pre-compute vertex indices as tensor for efficiency
        self._vertex_indices = torch.tensor(
            [self._atom_idxs[k] for k in self._atom_idxs_keys],
            dtype=torch.long,
            device=self._device
        )
        
        # Load KDE model
        kde_manager = KDEManager()
        self._kde_pdf = kde_manager.get_kde(self._kde_file, self._device)
    
    # properties
    @property
    def method(self) -> str:
        return self._method
    @property
    def kde_pdf(self) -> KernelDensity:
        return self._kde_pdf
    @property
    def atom_idxs(self) -> AtomIndexMap:
        return self._atom_idxs
    @property
    def vertex_indices(self) -> torch.Tensor:
        """Returns a tensor of shape [4] that contains the global 
        atom indices of the 4 vertices of the relevant tetrahedron."""
        return self._vertex_indices
    @property
    def temp(self) -> float:
        return self._temp
    @property
    def weight(self) -> float:
        return self._weight
    @property
    def offset(self) -> float:
        return self._offset
    @property
    def resonance_key(self) -> Optional[ResonanceKey]:
        return self._resonance_key
    @property
    def device(self) -> torch.device:
        return self._device
    
    # class method getters (property + classmethod has been deprecated)
    @classmethod
    def get_method(cls) -> str:
        return cls._method
    @classmethod
    def get_atom_idxs_keys(cls) -> AtomIndexKeys:
        return cls._atom_idxs_keys
    @classmethod
    def get_kde_file(cls) -> str:
        return cls._kde_file
    @classmethod
    def get_kde_pdf(cls, device: torch.device) -> KernelDensity:
        kde_manager = KDEManager()
        return kde_manager.get_kde(cls._kde_file, device)
    
    # methods
    @final
    def _compute_distances(self, 
                           positions: torch.Tensor, # [n_batch, n_atoms, 3]
                           vertex_indices: torch.Tensor, # [4]
                           ) -> torch.Tensor: # shape [n_batch, 6]
        """
        Compute the distances between the vertices of the tetrahedron.
        This is a helper function for _eval_loss.
        """
        # Extract tetrahedral vertex positions using pre-computed indices
        vertex_positions = positions[:, vertex_indices, :]
        
        # Stack atom positions for vectorized distance calculation
        v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
        
        atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
        atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
        
        # Calculate pairwise distances
        dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1) # shape: (batch_size, 6)
        return dists
    
    @final
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate loss for a batch of atom positions (in Angstroms).
        """
        # Compute distances
        dists = self._compute_distances(positions, self._vertex_indices)
        
        # Evaluate log probability using KDE
        logP = self.kde_pdf.score_samples(dists)
        
        # Convert to energy using Boltzmann relation
        return -KB * self._temp * logP
    
    @final
    def __call__(self, positions: torch.Tensor, _suppress_warnings: bool = False) -> torch.Tensor:
        """Compute total loss from atom positions in Angstroms."""
        if not _suppress_warnings:
            warnings.warn("ChemicalLoss instances should not be called directly. Use ChemicalLossHandler instead")
        loss = self._eval_loss(positions)
        return loss * self._weight + self._offset
    
    @classmethod
    def get_loss_instances(cls: type['ChemicalLoss'], 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,

                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['ChemicalLoss', ...]:
        """
        Identify valid atom configurations for this loss in a structure.
        Must be implemented by subclasses.
        
        Args:
            traj: MDTraj trajectory for analysis and atom identification
            atom_indexes_dict: Dictionary mapping (residue_idx, atom_name) to global atom indices (int)
            weight: Weight of the loss (float)
            offset: Offset of the loss (float)
            temp: Temperature of the loss (float)
            device: Device to use for the loss (torch.device)
            
        Returns:
            Tuple of ChemicalLoss instances.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement get_loss_instances")
    
    # --- methods to help subclasses build their own get_loss_instances methods ---
    @staticmethod
    def _get_ordered_residue_pairs(traj: md.Trajectory, 
                                   res1_name: str, 
                                   res2_name: str,
                                   ) -> Tuple[Tuple[Any, Any], ...]:
        """
        Returns a list of tuples of the form (res1, res2). Largely for use in `get_loss_instances`. 
        Always avoids duplicate pairs and always avoids pairing a residue with itself.
        """
        pairs = []
        if res1_name == res2_name:
            res_list = [res for res in traj.topology.residues if res.name == res1_name]
            for i in range(len(res_list)):
                # Use combinations (i < j) to avoid duplicates when same residue type
                for j in range(i+1, len(res_list)):
                    pairs.append((res_list[i], res_list[j]))
        else:
            res1_list = [res for res in traj.topology.residues if res.name == res1_name]
            res2_list = [res for res in traj.topology.residues if res.name == res2_name]
            # Generate all ordered pairs
            for res1 in res1_list:
                for res2 in res2_list:
                    if res1.index != res2.index:  # Avoid pairing a residue with itself
                        pairs.append((res1, res2))
        return tuple(pairs)
    
    @classmethod
    def _get_donor_acceptor_linkages(
        cls,

        traj: md.Trajectory,
        atom_indexes_dict: AtomIndexDict,

        donor_resname: str,
        acceptor_resname: str,
        donor_atom_groups: Dict[str, Sequence[str]],
        acceptor_atom_groups: Dict[str, Sequence[str]],

        weight: float,
        offset: float,
        temp: float,
        device: torch.device,
    ) -> Tuple['ChemicalLoss', ...]:
        """
        Returns a tuple of ChemicalLoss instances for a donor-acceptor linkage, and given the required atom names.

        Here is a high level overview of the method:
        Main Method:
        ├── Validate inputs
        ├── Get residue pairs  
        ├── For each pair:
        │   ├── Find donor atoms
        │   ├── Find acceptor atoms  
        │   ├── Skip if incomplete
        │   ├── Within a pair, skip any tetrahedra that have duplicate atoms for the different vertexes
        │   └── Generate all combinations
        └── Return results

        Args:
            traj: MDTraj trajectory for analysis and atom identification
            atom_indexes_dict: Dictionary mapping (residue_idx, atom_name) to global atom indices (int)
            donor_resname: Name of the donor residue
            acceptor_resname: Name of the acceptor residue
            donor_atom_groups: Dictionary mapping atom names to a sequence of atom names
            acceptor_atom_groups: Dictionary mapping atom names to a sequence of atom names
            weight: Weight of the loss
            offset: Offset of the loss
            temp: Temperature of the loss
            device: Device to use for the loss
        """
        # Validate inputs
        cls._validate_donor_acceptor_inputs(donor_atom_groups, acceptor_atom_groups)
        
        chem_losses = []

        # Get all valid donor-acceptor residue pairs
        pairs = ChemicalLoss._get_ordered_residue_pairs(traj, donor_resname, acceptor_resname)
        
        # Process each donor-acceptor residue pair to create ChemicalLoss instances
        for donor_res, acceptor_res in pairs:
            donor_idx = donor_res.index
            acceptor_idx = acceptor_res.index
            
            # Find all valid atoms for this residue pair
            donor_atoms = cls._find_valid_atoms_for_residue(
                donor_idx, donor_atom_groups, atom_indexes_dict
            )
            acceptor_atoms = cls._find_valid_atoms_for_residue(
                acceptor_idx, acceptor_atom_groups, atom_indexes_dict
            )
            
            # Skip if either residue is missing required atoms
            if not donor_atoms or not acceptor_atoms:
                continue
            
            # Combine donor and acceptor atoms and generate all combinations
            available_atoms = {**donor_atoms, **acceptor_atoms}
            resonance_key = (cls._method, frozenset([donor_idx, acceptor_idx]))
            
            # Generate ChemicalLoss instances for all atom combinations
            instances = cls._generate_loss_instances_for_pair(
                available_atoms, resonance_key, weight, offset, temp, device
            )
            chem_losses.extend(instances)
        
        return tuple(chem_losses)
    
    @classmethod
    def _validate_donor_acceptor_inputs(
        cls,
        donor_atom_groups: Dict[str, Sequence[str]],
        acceptor_atom_groups: Dict[str, Sequence[str]]
    ) -> None:
        """
        Validates the inputs to the _get_donor_acceptor_linkages method.
        """
        donor_keys = set(donor_atom_groups.keys())
        acceptor_keys = set(acceptor_atom_groups.keys())
        common_keys = donor_keys.intersection(acceptor_keys)
        all_keys = donor_keys.union(acceptor_keys)

        if len(all_keys) != 4:
            raise ValueError(f"Total number of keys in donor_atom_groups and acceptor_atom_groups must be exactly 4. "
                            f"Got {len(all_keys)} (donor: {len(donor_keys)}, acceptor: {len(acceptor_keys)})")
        if all_keys != set(cls._atom_idxs_keys):
            raise ValueError(f"All keys in donor_atom_groups and acceptor_atom_groups must be exactly the same as the "
                             f"ChemicalLoss class's atom_idxs_keys. Got {all_keys} (expected {cls._atom_idxs_keys})")
        if common_keys:
            raise ValueError(f"donor_atom_groups and acceptor_atom_groups must share no keys in common. "
                            f"Found common keys: {common_keys}")
    
    @classmethod
    def _find_valid_atoms_for_residue(
        cls,
        residue_idx: int,
        atom_groups: Dict[str, Sequence[str]],
        atom_indexes_dict: AtomIndexDict
    ) -> Optional[Dict[str, list[int]]]:
        """
        Find all valid atoms for a given residue and atom groups.
        
        Args:
            residue_idx: Index of the residue to search in
            atom_groups: Dictionary mapping atom keys (eg 'N1', 'C2') used in the subclasses to possible pdb atom names
            atom_indexes_dict: Dictionary mapping (residue_idx, atom_name) to global atom indices
            
        Returns:
            Dictionary mapping atom keys to lists of valid atom indices, or None if any key has no valid atoms
        """
        available_atoms: Dict[str, list[int]] = {}
        
        for atom_key, atom_names in atom_groups.items():
            valid_atom_indices = []
            for atom_name in atom_names:
                atom_lookup_key = (residue_idx, atom_name)
                if atom_lookup_key in atom_indexes_dict:
                    valid_atom_indices.append(atom_indexes_dict[atom_lookup_key])
            
            if not valid_atom_indices:
                # No valid atoms found for this key, return None to indicate failure
                return None
            available_atoms[atom_key] = valid_atom_indices
        
        return available_atoms
    
    @classmethod
    def _generate_loss_instances_for_pair(
        cls,
        available_atoms: Dict[str, list[int]],
        resonance_key: ResonanceKey,
        weight: float,
        offset: float,
        temp: float,
        device: torch.device
    ) -> list['ChemicalLoss']:
        """
        Generate all possible ChemicalLoss instances for a given set of available atoms.
        Automatically skips any combinations that have duplicate atoms for different keys.
        
        Args:
            available_atoms: Dictionary mapping atom keys to lists of valid atom indices
            resonance_key: Resonance key identifying this donor-acceptor pair
            weight: Weight of the loss
            offset: Offset of the loss
            temp: Temperature of the loss
            device: Device to use for the loss
            
        Returns:
            List of ChemicalLoss instances for all possible atom combinations
        """
        
        # Get the atom keys in a consistent order
        atom_keys = list(available_atoms.keys())
        atom_options = [available_atoms[key] for key in atom_keys]
        
        instances: List['ChemicalLoss'] = []
        
        # Generate a ChemicalLoss instance for each possible atom combination
        for atom_indices_combo in product(*atom_options):
            if len(set(atom_indices_combo)) != len(atom_indices_combo):
                continue
            # Build atom_idxs dict for this specific combination
            atom_idxs = {
                atom_keys[i]: atom_indices_combo[i] 
                for i in range(len(atom_keys))
            }
            
            # Create ChemicalLoss instance for this specific tetrahedron
            loss_instance = cls(
                atom_idxs=atom_idxs,
                temp=temp,
                weight=weight,
                offset=offset,
                resonance_key=resonance_key,
                device=device
            )
            instances.append(loss_instance)
        
        return instances