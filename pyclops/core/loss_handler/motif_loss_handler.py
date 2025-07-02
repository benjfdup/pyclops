import warnings
from typing import Optional

import torch

from ...utils.utils import motif_loss
from .loss_handler import LossHandler

class MotifLossHandler(LossHandler):
    '''
    A loss handler that computes structural deviation between input positions and a reference motif.
    
    This class implements a rotationally and translationally invariant structural deviation loss
    using the Kabsch algorithm for optimal alignment. The loss can be configured to ignore small
    deviations (tolerance) and to return either RMSD or MSD.
    
    Parameters
    ----------
    units_factor : float
        Conversion factor from input position units to angstroms.
    motif : torch.Tensor
        Reference structure to compare against, of shape [n_atoms, 3].
    motif_units_factor : float, optional
        Conversion factor from motif units to angstroms. Default is equal to units_factor.
    tolerance : float, optional
        No-penalty range around target positions in motif units. Default is 0.0.
    squared : bool, optional
        If True, returns MSD instead of RMSD. Default is False.
    allow_flips : bool, optional
        If True, allows the Kabsch algorithm to flip the coordinate system. Default is False.
    '''

    def __init__(self,
                 # standard args
                 units_factor: float,
                 
                 # motif args
                 motif: torch.Tensor,
                 motif_units_factor: Optional[float] = None,
                 tolerance: float = 0.0,
                 squared: bool = False,
                 allow_flips: bool = False,
                 ):
        super().__init__(units_factor)
        self.__validate_init_inputs(units_factor, motif, motif_units_factor, tolerance, squared, allow_flips)

        # Set default motif_units_factor to match units_factor if not provided
        if motif_units_factor is None:
            motif_units_factor = units_factor
        self._motif = motif * motif_units_factor # convert to angstroms
        self._tolerance = tolerance * motif_units_factor  # convert to angstroms
        self._squared = squared
        self._allow_flips = allow_flips
    
    def __validate_init_inputs(self,
                               units_factor: float,
                               motif: torch.Tensor,
                               motif_units_factor: Optional[float],
                               tolerance: float,
                               squared: bool,
                               allow_flips: bool,
                               ) -> None:
        '''Validate the inputs to the MotifLossHandler.'''
        if motif_units_factor != units_factor:
            warnings.warn(
                f"Motif units factor ({motif_units_factor}) does not match units factor ({units_factor}). "
                "This will still work, but you should take care that units are handled correctly throughout.")
        if not isinstance(motif, torch.Tensor):
            raise TypeError(f"Motif must be a torch.Tensor. Got {type(motif)}.")
        if not isinstance(motif_units_factor, (float, int)):
            raise TypeError(f"Motif units factor must be a float or int. Got {type(motif_units_factor)}.")
        if not isinstance(tolerance, (float, int)):
            raise TypeError(f"Tolerance must be a float or int. Got {type(tolerance)}.")
        if not isinstance(squared, bool):
            raise TypeError(f"Squared must be a bool. Got {type(squared)}.")
        if not isinstance(allow_flips, bool):
            raise TypeError(f"Allow flips must be a bool. Got {type(allow_flips)}.")

    @property
    def tolerance(self) -> float:
        """Tolerance in angstroms."""
        return self._tolerance
    
    @property
    def device(self) -> torch.device:
        """Device of the motif tensor."""
        return self._motif.device

    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute the structural deviation loss between input positions and the motif.

        Parameters
        ----------
        positions : torch.Tensor
            Input positions of shape [batch_size, n_atoms, 3] in angstroms.

        Returns
        -------
        torch.Tensor
            Loss values of shape [batch_size, ] in angstroms.
        """
        if positions.device != self.device:
            raise ValueError(f"Positions and motif are on different devices. "
                             f"Positions device: {positions.device}, motif device: {self.device}")
            
        return motif_loss(positions, 
                          self._motif, 
                          tolerance=self._tolerance, 
                          squared=self._squared, 
                          allow_flips=self._allow_flips)