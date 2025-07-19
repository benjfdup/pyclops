import torch

from ..loss_handler.loss_handler import LossHandler


class GyrationLossHandler(LossHandler):
    def __init__(self, units_factor: float = 1.0, squared: bool = 1.0):
        self._squared = squared
        super().__init__(units_factor)

    @property
    def squared(self) -> bool:
        return self._squared
    
    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute a loss based on the radius of gyration of atom positions in a generated peptide.
        
        Parameters
        ----------
        positions : torch.Tensor of shape [batch_size, n_atoms, 3]
            A tensor representing 3D atom positions for a batch of peptides.
            These positions are in angstroms.
        
        Returns
        -------
        loss : torch.Tensor of shape [batch_size,]
            A tensor containing the radius of gyration (or squared Rg if self.squared=True)
            as the loss value for each item in the batch.
        """
        center_of_mass = positions.mean(dim=1, keepdim=True)  # [batch_size, 1, 3]
        squared_distances = torch.sum((positions - center_of_mass) ** 2, dim=2)  # [batch_size, n_atoms]
        mean_squared_distances = torch.mean(squared_distances, dim=1)  # [batch_size]
        
        # Avoid the conditional branch in the computation graph for better performance
        if self.squared:
            return mean_squared_distances
        else:
            return torch.sqrt(mean_squared_distances)