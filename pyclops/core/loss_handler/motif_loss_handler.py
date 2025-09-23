import warnings
from typing import Optional, Tuple, Dict

import torch
import mdtraj as md

from ...utils.utils import motif_loss
from .loss_handler import LossHandler

AtomKey = Tuple[int, str]  # (residue_idx, atom_name)
AtomIndexDict = Dict[AtomKey, int]

class MotifLossHandler(LossHandler):
    """
    A loss handler that computes structural deviation between input positions and a reference motif.
    
    This class implements a rotationally and translationally invariant structural deviation loss
    using the Kabsch algorithm for optimal alignment. The loss can be configured to ignore small
    deviations (tolerance) and to return either RMSD or MSD.
    """

    def __init__(self, 
                 motif_dictionary: Dict[Tuple[int, str], torch.Tensor],
                 trajectory: md.Trajectory,
                 tolerance: float = 0.0,
                 units_factor: float = 1.0,
                 squared:bool = True,
                 device: Optional[torch.device] = None,
                 ):
        super().__init__(units_factor)
        self._squared = squared
        self._validate_inputs(motif_dictionary, trajectory, device)
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._motif_dictionary = motif_dictionary
        self._tolerance = tolerance
        self._ordered_keys = sorted(motif_dictionary.keys())
        motif_list = []
        for key in self._ordered_keys:
            motif_list.append(motif_dictionary[key])
        self._motif = torch.stack(motif_list, device=self._device)
        self._md_topology = trajectory.topology
        self._atom_indexes_dict = self._generate_atom_indexes_dict(self._md_topology)
        
        # Create tensor of atom indices for Kabsch conditioning
        idxs_list = []
        for key in self._ordered_keys:
            idxs_list.append(self._atom_indexes_dict[key])
        self._idxs_tensor = torch.tensor(idxs_list, dtype=torch.long, device=self._device)

    def _validate_inputs(self, 
                        motif_dictionary: Dict[Tuple[int, str], torch.Tensor],
                        trajectory: md.Trajectory,
                        device: Optional[torch.device]) -> None:
        """
        Simple validation for MotifLossHandler inputs.
        """
        # Validate trajectory
        if not isinstance(trajectory, md.Trajectory):
            raise ValueError("trajectory must be an instance of mdtraj.Trajectory")
        if trajectory.n_atoms == 0:
            raise ValueError("trajectory must have at least one atom")
        if trajectory.n_residues == 0:
            raise ValueError("trajectory must have at least one residue")
        if trajectory.n_chains != 1:
            raise ValueError("trajectory must have only one chain")
        
        # Validate motif dictionary
        if not isinstance(motif_dictionary, dict):
            raise ValueError("motif_dictionary must be a dictionary")
        if len(motif_dictionary) == 0:
            raise ValueError("motif_dictionary must not be empty")
        
        # Validate device
        if device is not None and not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")

    @staticmethod
    def _generate_atom_indexes_dict(topology: md.Topology) -> AtomIndexDict:
        indices = {}
        for residue in topology.residues:
            for atom in residue.atoms:
                indices[(residue.index, atom.name)] = atom.index
        return indices

    def _eval_loss(self, 
                   positions: torch.Tensor, # shape: [n_batch, n_atoms, 3] (full positions)
                   ) -> torch.Tensor: # shape: [n_batch, ]
        pos = positions[:, self._idxs_tensor, :]

        return motif_loss(pos, self._motif, device=self._device, squared=self._squared, tolerance=self._tolerance)