from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np

class BaseRelaxer(ABC):
    def __init__(self, units_factor: float, pdb_filepath: str):
        self._units_factor = units_factor
        self._pdb_filepath = pdb_filepath

    @abstractmethod
    def relax(self, structure: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")