from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np

class BaseRelaxer(ABC):
    def __init__(self, units_factor: float, pdb_filepath: str):
        self._units_factor = units_factor
        self._pdb_filepath = pdb_filepath

    @property
    def units_factor(self) -> float:
        return self._units_factor
    
    @property
    def pdb_filepath(self) -> str:
        return self._pdb_filepath
    
    def _convert_to_angstroms(self, coordinates: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Convert input coordinates to angstroms"""
        return coordinates * self.units_factor
    
    def _convert_from_angstroms(self, coordinates: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Convert coordinates from angstroms to original units"""
        return coordinates / self.units_factor

    @abstractmethod
    def relax(self, structure: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")