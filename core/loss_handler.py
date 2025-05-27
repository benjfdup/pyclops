from abc import ABC, abstractmethod
from typing import final
import torch

class LossHandler(ABC):
    '''
    Abstract class whose subclasses can wrap losses which do not vary with time.
    '''

    def __init__(self, 
                 units_factor: float, # we may need to make this be a tensor
                 ):
        
        self._units_factor = units_factor # x * units_factor -> x in angstroms

    @property
    def units_factor(self) -> float:
        return self._units_factor
    @units_factor.setter
    def units_factor(self, new_factor):
        self._units_factor = new_factor

    @abstractmethod
    def _eval_loss(self, positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the loss for atom positions in a generated peptide.

        Parameters
        ----------
        positions : torch.Tensor of shape [batch_size, n_atoms, 3]
            A tensor representing 3D atom positions for a batch of peptides.
            These positions MUST be in angstroms.

        Returns
        -------
        loss : torch.Tensor of shape [batch_size,]
            A tensor containing the loss value for each item in the batch.
        """
        raise NotImplementedError('Implement this in the relevant subclass')
    
    @final
    def _convert_positions(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Converts the input positions from their input units to angstroms
        '''
        t_pos = positions * self._units_factor
        return t_pos
    
    def __call__(self, positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Evaluate the loss function for a batch of atom positions.

        Parameters
        ----------
        positions : torch.Tensor of shape (batch_size, n_atoms, 3)
            A tensor representing 3D atom positions for a batch of peptides.

        Returns
        -------
        loss : torch.Tensor of shape (batch_size,)
            A tensor containing the loss value for each item in the batch.
        """
        positions_ang = self._convert_positions(positions)
        return self._eval_loss(positions_ang)