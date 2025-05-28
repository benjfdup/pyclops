"""
All-atom scoring functions for biomolecules using PyRosetta.
"""

from typing import Dict, List, Optional
from rosetta import *
init()

class RosettaScorer:
    def __init__(self, scorefxn_name: str = 'ref2015'):
        """
        Initialize the PyRosetta scorer.
        
        Args:
            scorefxn_name: Name of the scoring function to use
        """
        self.scorefxn_name = scorefxn_name
        self.scorefxn = self._get_scorefxn()
    
    def _get_scorefxn(self):
        """Get the scoring function."""
        if self.scorefxn_name == 'ref2015':
            return get_fa_scorefxn()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scorefxn_name}")
    
    def calculate_total_energy(self, pose: Pose) -> float:
        """
        Calculate the total energy of a pose using the standard scoring function.
        
        Args:
            pose: PyRosetta Pose object containing the biomolecule
            
        Returns:
            float: Total energy score
        """
        return self.scorefxn(pose)
    
    def get_energy_components(self, pose: Pose) -> Dict[str, float]:
        """
        Get detailed energy components for a pose.
        
        Args:
            pose: PyRosetta Pose object containing the biomolecule
            
        Returns:
            Dict[str, float]: Dictionary of energy component names and their values
        """
        self.scorefxn(pose)
        
        energy_map = {}
        for term in self.scorefxn.get_nonzero_weighted_scoretypes():
            energy_map[term.name] = self.scorefxn.get_weighted_score(pose, term)
        
        return energy_map
    
    def calculate_interface_energy(self, pose: Pose, chain_ids: List[int]) -> float:
        """
        Calculate the interface energy between specified chains.
        
        Args:
            pose: PyRosetta Pose object containing the biomolecule
            chain_ids: List of chain IDs to consider for interface calculation
            
        Returns:
            float: Interface energy score
        """
        # Create a copy of the pose for interface calculation
        interface_pose = Pose()
        interface_pose.assign(pose)
        
        # Calculate interface energy
        interface_energy = 0.0
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                interface_energy += self.scorefxn.get_interface_energy(interface_pose, chain_ids[i], chain_ids[j])
        
        return interface_energy
    
    def calculate_solvation_energy(self, pose: Pose) -> float:
        """
        Calculate the solvation energy of a pose.
        
        Args:
            pose: PyRosetta Pose object containing the biomolecule
            
        Returns:
            float: Solvation energy score
        """
        self.scorefxn(pose)
        return self.scorefxn.get_weighted_score(pose, fa_sol)
    
    def calculate_hbond_energy(self, pose: Pose) -> float:
        """
        Calculate the hydrogen bond energy of a pose.
        
        Args:
            pose: PyRosetta Pose object containing the biomolecule
            
        Returns:
            float: Hydrogen bond energy score
        """
        self.scorefxn(pose)
        return (self.scorefxn.get_weighted_score(pose, hbond_sr_bb) + 
                self.scorefxn.get_weighted_score(pose, hbond_lr_bb)) 