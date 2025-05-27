import torch
import warnings

from ..core.loss_handler import LossHandler
from ..utils.utils import motif_loss


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
        Conversion factor from motif units to angstroms. Default is 1.0.
    tolerance : float, optional
        No-penalty range around target positions in motif units. Default is 0.0.
    squared : bool, optional
        If True, returns MSD instead of RMSD. Default is False.
    '''

    def __init__(self,
                 # standard args
                 units_factor: float,
                 
                 # motif args
                 motif: torch.Tensor,
                 motif_units_factor: float = 1.0,
                 tolerance: float = 0.0,
                 squared: bool = False,
                 ):
        self._motif = motif * motif_units_factor
        self._motif_units_factor = motif_units_factor
        self._tolerance = tolerance * motif_units_factor  # Convert to angstroms
        self._squared = squared

        if self._motif_units_factor != units_factor:
            warnings.warn(
                f"Motif units factor ({self._motif_units_factor}) does not match units factor ({units_factor}). "
                "This will still work, but you should take care that units are handled correctly throughout."
                )
            
        super().__init__(units_factor)

    @property
    def motif_units_factor(self) -> float:
        '''Conversion factor from motif units to angstroms.'''
        return self._motif_units_factor

    @motif_units_factor.setter
    def motif_units_factor(self, motif_units_factor: float) -> None:
        '''Update the motif units factor and rescale stored values.'''
        original_motif = self._motif / self._motif_units_factor
        new_motif = original_motif * motif_units_factor
        self._motif_units_factor = motif_units_factor
        self._tolerance = self._tolerance * motif_units_factor
        self._motif = new_motif

    @property
    def tolerance(self) -> float:
        '''Tolerance in motif units (before conversion to angstroms).'''
        return self._tolerance / self._motif_units_factor

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        '''Set tolerance in motif units.'''
        self._tolerance = tolerance * self._motif_units_factor

    def motif_to(self, device: torch.device) -> None:
        '''
        Move the motif tensor to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device for the motif tensor.
        '''
        self._motif = self._motif.to(device)

    def _eval_loss(self, positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        '''
        Compute the structural deviation loss between input positions and the motif.

        Parameters
        ----------
        positions : torch.Tensor
            Input positions of shape [batch_size, n_atoms, 3] in angstroms.

        Returns
        -------
        torch.Tensor
            Loss values of shape [batch_size,] in angstroms.
        '''
        if self._motif.device != positions.device:
            warnings.warn(
                f"Motif and positions are on different devices. "
                "Moving motif to positions device."
                )
            self.motif_to(positions.device)
            
        return motif_loss(positions, self._motif, tolerance=self._tolerance, squared=self._squared)