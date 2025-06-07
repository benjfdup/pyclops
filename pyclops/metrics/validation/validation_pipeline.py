from typing import Union
import torch
import numpy as np
from ..scoring.base_scorer import BaseScorer
from scipy.stats import wasserstein_distance

class ValidationPipeline():
    """
    This class is used to validate the performance of a model.
    """
    valid_metrics = ["w1"] # only w1 for now

    def __init__(self, 
                 scorer: BaseScorer,
                 distance_metric: str,
                 ):
        self._scorer = scorer
        self._distance_metric = self._get_distance_metric(distance_metric)
        self._validation_set = None

    @classmethod
    def _get_distance_metric(cls, distance_metric: str) -> str:
        d_met = distance_metric.lower()
        if d_met in cls.valid_metrics:
            return d_met
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}. Valid metrics are: {cls.valid_metrics}")
        
    def initialize_validation_set(self, 
                                  coordinates: Union[torch.Tensor, np.ndarray],
                                  ):
        """
        Initialize the validation set.
        """
        energies = self._scorer.calculate_energy(coordinates)
        
        # Ensure energies is always a numpy array for scipy compatibility
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy()
        elif not isinstance(energies, np.ndarray):
            energies = np.array(energies)
        
        # Ensure energies is 1D for wasserstein_distance
        energies = energies.flatten()
            
        self._validation_set = energies # [n_batch, ]

    def validate(self, 
                 coordinates: Union[torch.Tensor, np.ndarray],
                 ) -> float:
        """
        Compute the distance between sample coordinates and the validation set.
        
        Args:
            coordinates: Sample coordinates to compare against validation set
            
        Returns:
            Distance metric (e.g., Wasserstein distance) between sample and validation energies
        """
        if self._validation_set is None:
            raise ValueError("Validation set not initialized. Call initialize_validation_set first.")
        
        sample_energies = self._scorer.calculate_energy(coordinates)
        if isinstance(sample_energies, torch.Tensor):
            sample_energies = sample_energies.detach().cpu().numpy()
        elif not isinstance(sample_energies, np.ndarray):
            sample_energies = np.array(sample_energies)
        
        # Ensure sample_energies is 1D for wasserstein_distance
        sample_energies = sample_energies.flatten()
        
        if self._distance_metric == "w1":
            distance = wasserstein_distance(self._validation_set, sample_energies)
        else:
            raise ValueError(f"Invalid distance metric: {self._distance_metric}. Valid metrics are: {self.valid_metrics}")
        
        return distance

    # we likely want she user to pass in a series of points, [n_batch, n_atoms, 3]
    # which will consitute our "validation set". We will then compute an energy for
    # each point in the validation set --> [n_batch, ]
    
    # then we will provide a function in which the user can pass a new series of points, 
    # [n_batch, n_atoms, 3], which we will compute an energy for --> [n_batch, ]
    # and which we will then compute a wasserstein distance between the new energy and the 
    # energy of the validation set.

    # we will then return this wasserstein distance.