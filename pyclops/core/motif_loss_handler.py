import torch

from ..core.loss_handler import LossHandler
from ..utils.utils import motif_loss


class MotifLossHandler(LossHandler):
    '''
    Class to handle RMSD motif losses.
    '''

    def __init__(self,
                 motif: torch.Tensor,
                 units_factor: float,
                 motif_units_factor: float = 1.0, # motif * motif_units_factor = motif_in_angstroms
                 ):
        self._motif = motif * motif_units_factor
        self._motif_units_factor = motif_units_factor # this will be stored simply for future reference.

        super().__init__(units_factor)

    @property
    def motif_units_factor(self) -> float:
        return self._motif_units_factor
    @motif_units_factor.setter
    def motif_units_factor(self, motif_units_factor: float) -> None:
        # raise some warning here about not really being supposed to do this
        original_motif = self._motif / self._motif_units_factor
        new_motif = original_motif * motif_units_factor
        self._motif_units_factor = motif_units_factor
        self._motif = new_motif
    # We also need a method (like motif_to or smthng) which moves the motif to the desired device.

    def motif_to(self, device: torch.device) -> None:
        """
        Move the motif tensor to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the motif tensor to.
        """
        self._motif = self._motif.to(device)

    def _eval_loss(self, positions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Compute the RMSD loss between the input positions and the stored motif.

        Parameters
        ----------
        positions : torch.Tensor of shape [batch_size, n_atoms, 3]
            A tensor representing 3D atom positions for a batch of peptides.
            These positions MUST be in angstroms.

        Returns
        -------
        loss : torch.Tensor of shape [batch_size,]
            A tensor containing the RMSD loss value for each item in the batch.
        """
        # Ensure motif is on the same device as positions
        if self._motif.device != positions.device:
            #self.motif_to(positions.device) 
            raise RuntimeError(f"Device mismatch: {self._motif} is on {self._motif.device}, but {positions} is on {positions.device}.")
            
        return motif_loss(positions, self._motif)