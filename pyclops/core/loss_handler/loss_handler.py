from abc import ABC, abstractmethod
import torch

class LossHandler(ABC):
    """
    Base class for loss handlers.
    This class is designed to be immutable after initialization.
    """
    
    def __init__(self, units_factor: float):
        """Initialize a LossHandler instance.
        Args:
            units_factor: The factor to convert the units of the positions to Angstroms. 
            positions * units_factor = Angstroms
        """
        self._validate_inputs(units_factor)
        self._units_factor = units_factor

    def _validate_inputs(self, units_factor: float) -> None:
        """Validate the inputs to the LossHandler."""
        if units_factor <= 0.0:
            raise ValueError(f"units_factor must be positive. Got {units_factor}")
    
    @property
    def units_factor(self) -> float:
        return self._units_factor
    
    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total loss from atom positions."""
        # Convert positions to Angstroms if needed
        positions_ang = positions * self._units_factor
        
        # Call the scripted method through the class instance
        return self._eval_loss(positions_ang) # shape: [n_batch, ]
    
    @abstractmethod
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the loss for a batch of atom positions.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement method _eval_loss")