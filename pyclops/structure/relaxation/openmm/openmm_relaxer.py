import torch
import numpy as np
from typing import Union
from ..base_relaxer import BaseRelaxer


class OpenMMRelaxer(BaseRelaxer):
    def __init__(self, units_factor: float, pdb_filepath: str):
        super().__init__(units_factor, pdb_filepath)

    def relax(self, structure: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError("OpenMMRelaxer is not implemented yet")