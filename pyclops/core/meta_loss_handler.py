from typing import List, Union
import warnings

import torch

from ..core.loss_handler import LossHandler


#### NEED TO ADD SUPPORT FOR COEFFICIENTS WHICH VARY WITH TIME (LOSS-COEFFs)... ####
class MetaLossHandler(LossHandler):
    '''
    A `LossHandler` which wraps other LossHandlers for its `_eval_loss` method.

    Note unit_conversion is handled in the `MetaLossHandler`, rather than in
    subordinate `LossHandler`s
    '''

    def __init__(self,
                 subordinates: List[LossHandler],
                 subordinate_factors: Union[List[float], torch.Tensor], # if a tensor, must be shape [len(subordinates)]
                 units_factor: float,
                 ):
        super().__init__(units_factor=units_factor)

        if len(subordinate_factors) != len(subordinates):
            raise ValueError('subordinate_factors must be the same length as subordinates')
        
        self._subordinates = subordinates
        
        # Convert subordinate_factors to tensor if it's a list
        if isinstance(subordinate_factors, list):
            self._subordinate_factors = torch.tensor(subordinate_factors, dtype=torch.float)
        else:
            self._subordinate_factors = subordinate_factors
        
        if torch.any(self._subordinate_factors < 0.0):
            warnings.warn(
                "[MetaLossHandler] Negative subordinate factors detected in MetaLossHandler. "
                "This may result in loss minimization working against the affected subordinate objectives.",
                UserWarning
            )
    
    def _eval_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Evaluate the combined loss by summing the weighted losses from all subordinate handlers.
        
        Returns:
            torch.Tensor: The combined weighted loss
        """
        total_loss = 0.0
        
        for i, handler in enumerate(self._subordinates):
            # Pass raw inputs to subordinates, but don't apply their unit conversion
            # as we'll handle unit conversion at the meta level
            raw_loss = handler._eval_loss(*args, **kwargs)
            weighted_loss = raw_loss * self._subordinate_factors[i]
            total_loss += weighted_loss
            
        return total_loss # [n_batch, ]
    
    def get_subordinate_losses(self, *args, **kwargs) -> List[torch.Tensor]:
        """
        Return individual losses from each subordinate handler before weighting.
        
        Returns:
            List[torch.Tensor]: List of unweighted losses from each subordinate handler
        """
        return [handler._eval_loss(*args, **kwargs) for handler in self._subordinates]
    
    def get_weighted_subordinate_losses(self, *args, **kwargs) -> List[torch.Tensor]:
        """
        Return individual losses from each subordinate handler after weighting.
        
        Returns:
            List[torch.Tensor]: List of weighted losses from each subordinate handler
        """
        return [handler._eval_loss(*args, **kwargs) * self._subordinate_factors[i] 
                for i, handler in enumerate(self._subordinates)]
    
    def __repr__(self) -> str:
        """
        String representation of the MetaLossHandler.
        
        Returns:
            str: String representation
        """
        subordinate_reprs = [f"{factor} * {type(handler).__name__}" 
                            for handler, factor in zip(self._subordinates, self._subordinate_factors)]
        return f"MetaLossHandler(units_factor={self._units_factor}, subordinates=[{', '.join(subordinate_reprs)}])"