from abc import ABC, abstractmethod
from typing import final, Dict, List, Optional, Set, Type, Union, Tuple
import torch
import mdtraj as md
from pathlib import Path

from ..utils.constants import UNITS_FACTORS_DICT

class LossHandler(ABC):
    """
    Base class for loss handlers.
    This class is designed to be immutable after initialization.
    """
    
    def __init__(self, units_factor: float):
        """Initialize a LossHandler instance."""
        self._units_factor = units_factor
    
    @property
    def units_factor(self) -> float:
        return self._units_factor
    
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total loss from atom positions."""
        # Convert positions to Angstroms if needed
        positions_ang = positions * self._units_factor
        
        # Call the scripted method through the class instance
        return self._eval_loss(positions_ang)
    
    @abstractmethod
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the loss for a batch of atom positions.
        Must be implemented by subclasses.
        """
        pass