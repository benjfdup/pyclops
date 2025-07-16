from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np
import mdtraj as md # for loading pdb files

# output energies in kj/mol
# output forces in kj/mol/angstrom

TensorLike = Union[torch.Tensor, np.ndarray]

class BaseScorer(ABC):
    """
    Base class for scoring functions. These will just wrap scoring functions from other packages.
    Really nothing to see here. Just some convenience functions.
    """
    def __init__(
            self,
            topology: md.Topology,
            units_factor: float, # input_coords * units_factor = coords_in_angstroms
            ):
        self._topology = topology
        self._units_factor = units_factor

    @property
    def units_factor(self) -> float:
        """
        The factor by which the positions are scaled.
        """
        return self._units_factor

    def _convert_to_angstroms(self,
                              coordinates: TensorLike,
                              ) -> TensorLike:
        """
        Convert input coordinates to angstroms using units_factor
        """
        return coordinates * self._units_factor

    @classmethod
    def from_pdb_file(cls,
                      pdb_file: str,
                      units_factor: float,
                      ) -> 'BaseScorer':
        """
        Create a BaseScorer from a PDB file.
        """
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor)

    @abstractmethod
    def calculate_energy(self,
                         coordinates: Union[torch.Tensor, np.ndarray], # shape: [n_batch, n_atoms, 3]
                         ) -> Union[torch.Tensor, np.ndarray]: # shape: [n_batch]
        """
        Calculate potential energy in kJ/mol.

        Args:
            coordinates: TensorLike, shape: [n_batch, n_atoms, 3]

        Returns:
            TensorLike, shape: [n_batch]
        """
        raise NotImplementedError("Subclass must implement this method")