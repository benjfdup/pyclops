from abc import ABC, abstractmethod
from typing import Union
import torch
import numpy as np

from .base_scorer import BaseScorer

class BaseValidationPipeline(ABC):
    def __init__(self, 
                 scorer: BaseScorer,
                 ):
        self._scorer = scorer

    @abstractmethod
    def validate(self, 
                 coordinates: Union[torch.Tensor, np.ndarray],
                 ) -> Union[torch.Tensor, np.ndarray]:
        pass