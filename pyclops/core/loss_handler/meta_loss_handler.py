from typing import Optional, Sequence
import warnings

import torch

from .loss_handler import LossHandler

class MetaLossHandler(LossHandler):
    """
    A `LossHandler` which wraps other `LossHandler`s for its `_eval_loss` method.

    Note unit_conversion is handled in the `MetaLossHandler`, rather than in
    subordinate `LossHandler`s
    """

    def __init__(self, 
                 units_factor: float, 
                 subordinates: Sequence[LossHandler],
                 subordinate_factors: Sequence[float],
                 device: Optional[torch.device] = None,
                 _suppress_warnings: bool = False,
                 ):
        super().__init__(units_factor)
        self.__validate_meta_loss_handler_init_inputs(units_factor, subordinates, subordinate_factors, _suppress_warnings)
        
        self._subordinates = subordinates
        self._subordinate_factors = torch.tensor(subordinate_factors, device=device)

    def __validate_meta_loss_handler_init_inputs(self,
                                                 units_factor: float,
                                                 subordinates: Sequence[LossHandler],
                                                 subordinate_factors: Sequence[float],
                                                 _suppress_warnings: bool,
                                                 ) -> None:
        """Validate the inputs to the `MetaLossHandler`."""
        if not isinstance(subordinates, Sequence):
            raise TypeError(f"Subordinates must be a sequence. Got {type(subordinates)}.")
        if not isinstance(subordinate_factors, Sequence):
            raise TypeError(f"Subordinate factors must be a sequence. Got {type(subordinate_factors)}.")
        if len(subordinate_factors) != len(subordinates):
            raise ValueError(f"Subordinate factors sequence must have length [len(subordinates)]. "
                             f"Got {len(subordinate_factors)}.")
        if any(factor < 0.0 for factor in subordinate_factors):
            if not _suppress_warnings:
                warnings.warn(
                    "[MetaLossHandler] Negative subordinate factors detected in MetaLossHandler. "
                    "This may result in loss minimization working against the affected subordinate objectives.",
                    UserWarning
                )

    def _eval_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the combined loss by summing the weighted losses from all subordinate handlers.

        Args:
            positions: Tensor of shape [batch_size, n_atoms, 3] representing the positions of the atoms.

        Returns:
            torch.Tensor: The combined weighted loss, of shape [batch_size, ]
        """
        n_batch = positions.shape[0]
        
        # Initialize combined loss tensor
        combined_loss = torch.zeros(n_batch, device=positions.device)
        
        # Evaluate loss from each subordinate handler and add weighted contribution
        for i, (subordinate, factor) in enumerate(zip(self._subordinates, self._subordinate_factors)):
            subordinate_loss = subordinate._eval_loss(positions)
            combined_loss += factor * subordinate_loss
            
        return combined_loss

