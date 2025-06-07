from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import torch
import numpy as np
import mdtraj as md # for loading pdb files

# output energies in kj/mol
# output forces in kj/mol/angstrom
# internally, distances are in angstroms

class BaseScorer(ABC):
    def __init__(self, 
                 pdb_path: str,
                 units_factor: float, # input_coords * units_factor = coords_in_angstroms
                 ):
        """
        Args:
            pdb_path: path to pdb file
            units_factor: factor to convert input coordinates to angstroms
        """
        self._pdb_path = pdb_path
        self._pdb = md.load_pdb(pdb_path)
        self._units_factor = units_factor

    @property
    def units_factor(self) -> float:
        """
        The factor by which the positions are scaled.
        """
        return self._units_factor

    def _convert_to_angstroms(self, 
                              coordinates: Union[torch.Tensor, np.ndarray], # shape: [n_batch, n_atoms, 3]
                              ) -> Union[torch.Tensor, np.ndarray]:
        """
        Convert input coordinates to angstroms
        """
        return coordinates * self._units_factor

    @abstractmethod
    def calculate_energy(self, 
                         coordinates: Union[torch.Tensor, np.ndarray], # shape: [n_batch, n_atoms, 3]
                         ) -> Union[torch.Tensor, np.ndarray]: # shape: [n_batch]
        
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    def calculate_forces(self, 
                         coordinates: Union[torch.Tensor, np.ndarray], # shape: [n_batch, n_atoms, 3]
                         ) -> Union[torch.Tensor, np.ndarray]: # shape: [n_batch, n_atoms, 3]
        raise NotImplementedError("Subclass must implement this method")