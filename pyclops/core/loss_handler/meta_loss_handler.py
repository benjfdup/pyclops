from typing import Optional, Sequence
import warnings

import torch

from .loss_handler import LossHandler
from ...utils.utils import soft_min

class MetaLossHandler(LossHandler):
    """
    A `LossHandler` which wraps other `LossHandler`s for its `_eval_loss` method.

    Note unit_conversion is handled in the `MetaLossHandler`, rather than in
    subordinate `LossHandler`s
    """
    allowed_mediations = ["sum", "avg", "softmin", "min"]

    def __init__(self, 
                 subordinates: Sequence[LossHandler],
                 subordinate_factors: Optional[Sequence[float]] = None,
                 mediation: str = "sum",
                 units_factor: float = 1.0, 
                 device: Optional[torch.device] = None,
                 alpha: float = -3.0,
                 _suppress_warnings: bool = False,
                 ):
        super().__init__(units_factor)
        self.__validate_meta_loss_handler_init_inputs(units_factor, subordinates, subordinate_factors, _suppress_warnings)
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._subordinates = subordinates
        if subordinate_factors is None:
            self._subordinate_factors = torch.ones(len(subordinates), device=self._device)
        else:
            self._subordinate_factors = torch.tensor(subordinate_factors, device=self._device)
        self._mediation = mediation
        self._alpha = alpha

    def __validate_meta_loss_handler_init_inputs(self,
                                                 units_factor: float,
                                                 subordinates: Sequence[LossHandler],
                                                 subordinate_factors: Sequence[float],
                                                 _suppress_warnings: bool,
                                                 ) -> None:
        """Validate the inputs to the `MetaLossHandler`."""
        if not isinstance(subordinates, Sequence):
            raise TypeError(f"Subordinates must be a sequence. Got {type(subordinates)}.")
        if subordinate_factors is not None:
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
        
        loss_list = []
        # Evaluate loss from each subordinate handler and add weighted contribution
        for i, (subordinate, factor) in enumerate(zip(self._subordinates, self._subordinate_factors)):
            subordinate_loss = subordinate._eval_loss(positions)
            loss_list.append(factor * subordinate_loss) # shape: [n_batch, ]

        loss_list = torch.stack(loss_list, dim=1) # shape: [n_batch, n_subordinates]

        if self._mediation == "sum":
            combined_loss = torch.sum(loss_list, dim=1) # shape: [n_batch, ]
        elif self._mediation == "avg":
            combined_loss = torch.mean(loss_list, dim=1) # shape: [n_batch, ]
        elif self._mediation == "softmin":
            combined_loss = soft_min(loss_list, alpha=self._alpha) # shape: [n_batch, ]
        elif self._mediation == "min":
            combined_loss = torch.min(loss_list, dim=1)[0] # shape: [n_batch, ]
            
        return combined_loss

