import torch
from typing import Dict

from ..utils.constants import KB
from ..utils.utils import soft_min


class CompiledEvalLoss(torch.nn.Module):
    """
    A compiled version of the chemical loss evaluation function.
    This class is designed to be used with torch.jit.script for improved performance.
    
    Parameters
    ----------
    resonance_groups : Dict
        Dictionary mapping resonance keys to lists of ChemicalLoss instances.
    temp : float
        Temperature in Kelvin for energy calculations.
    alpha : float
        Soft minimum parameter controlling the sharpness of the minimum.
    device : torch.device
        Device to use for computations.
    """
    
    def __init__(self, 
                 resonance_groups: Dict,
                 temp: float,
                 alpha: float,
                 device: torch.device):
        super().__init__()
        # Store all fixed parameters
        self._resonance_groups = resonance_groups
        self._temp = temp
        self._alpha = alpha
        self.KB = KB  # Boltzmann constant
        self._device = device  # Store device for KDE models
        
        # Pre-compute group information for JIT
        self._total_groups = len(resonance_groups)
        self._max_losses_per_group = max(len(losses) for losses in resonance_groups.values())
        
        # Pre-compute group sizes for each group
        self._group_sizes = torch.tensor(
            [len(losses) for losses in resonance_groups.values()],
            device=device
        )
        
        # Explicitly set the module's device
        self.to(device)
        
    @torch.jit.script_method
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the loss for a batch of atom positions.
        
        Parameters
        ----------
        positions : torch.Tensor
            Tensor of shape (batch_size, n_atoms, 3) containing atom positions in Angstroms.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size,) containing the combined loss values.
        """
        batch_size = positions.shape[0]
        
        # Pre-allocate with known shapes
        all_group_losses = torch.full(
            (batch_size, self._total_groups, self._max_losses_per_group),
            float('inf'),
            device=self._device
        )
        
        # Process each group
        for group_idx, (resonance_key, losses_in_group) in enumerate(self._resonance_groups.items()):
            group_size = self._group_sizes[group_idx]
            group_losses = torch.full(
                (batch_size, group_size),
                float('inf'),
                device=self._device
            )
            
            # Process each loss in the group
            for loss_idx, loss in enumerate(losses_in_group):
                vertex_indices = [loss.atom_idxs[key] for key in loss.atom_idxs_keys]
                vertex_positions = positions[:, vertex_indices, :]
                
                v0, v1, v2, v3 = vertex_positions[:, 0], vertex_positions[:, 1], vertex_positions[:, 2], vertex_positions[:, 3]
                
                atom_pairs_1 = torch.stack([v0, v0, v0, v1, v1, v2], dim=1)
                atom_pairs_2 = torch.stack([v1, v2, v3, v2, v3, v3], dim=1)
                
                dists = torch.linalg.vector_norm(atom_pairs_1 - atom_pairs_2, dim=-1)
                
                logP = loss.kde_pdf.score_samples(dists)
                energy = -self.KB * self._temp * logP
                scaled_energy = energy * loss.weight + loss.offset
                group_losses[:, loss_idx] = scaled_energy
            
            # Take minimum across all resonant structures in this group
            group_min = torch.min(group_losses, dim=1)[0]
            all_group_losses[:, group_idx, 0] = group_min
        
        # Apply soft minimum across all unique cyclization groups
        if self._total_groups == 1:
            return all_group_losses[:, 0, 0]
        else:
            group_losses = all_group_losses[:, :, 0]
            return soft_min(group_losses, alpha=self._alpha) 