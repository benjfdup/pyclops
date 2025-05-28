"""
Conversion utilities between PDB + atom coordinates and PyRosetta Pose objects.
"""

from typing import Tuple, Dict, List, Optional, Union
import torch
import numpy as np
import tempfile
import os
from rosetta import *
init()

class AtomCoordinateConverter:
    def __init__(self):
        """Initialize the converter with PyRosetta."""
        self.init_rosetta()
    
    @staticmethod
    def init_rosetta():
        """Initialize PyRosetta if not already initialized."""
        if not rosetta.basic.was_init_called():
            init()
    
    def pdb_to_pose(self, pdb_path: str) -> Tuple[Pose, Dict[str, torch.Tensor]]:
        """
        Convert a PDB file to a Pose object and extract atom coordinates.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            Tuple containing:
                - Pose object
                - Dictionary mapping atom names to their coordinates as torch tensors
        """
        pose = pose_from_pdb(pdb_path)
        atom_coords = self._extract_atom_coordinates(pose)
        return pose, atom_coords
    
    def coordinates_to_pose(self, 
                          pdb_path: str, 
                          new_coords: Union[torch.Tensor, np.ndarray],
                          chain_id: Optional[str] = None) -> Pose:
        """
        Create a Pose object from a PDB template and new atom coordinates.
        Creates a temporary PDB file with new coordinates and loads it into a Pose.
        
        Args:
            pdb_path: Path to the template PDB file
            new_coords: Tensor/array of shape (n_atoms, 3) containing new coordinates
            chain_id: Optional chain ID to modify. If None, modifies all chains.
            
        Returns:
            Pose object with new coordinates
        """
        # Convert coordinates to numpy if they're torch tensors
        if isinstance(new_coords, torch.Tensor):
            new_coords = new_coords.cpu().numpy()
            
        # Read the original PDB file
        with open(pdb_path, 'r') as f:
            pdb_lines = f.readlines()
            
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
            atom_idx = 0
            for line in pdb_lines:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    if chain_id is None or line[21] == chain_id:
                        # Update coordinates in the PDB line
                        x, y, z = new_coords[atom_idx]
                        line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                        atom_idx += 1
                temp_file.write(line)
            temp_path = temp_file.name
            
        try:
            # Create pose from the temporary PDB
            pose = pose_from_pdb(temp_path)
            return pose
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def _extract_atom_coordinates(self, pose: Pose) -> Dict[str, torch.Tensor]:
        """
        Extract atom coordinates from a Pose object into a dictionary of torch tensors.
        
        Args:
            pose: PyRosetta Pose object
            
        Returns:
            Dictionary mapping atom names to their coordinates as torch tensors
        """
        atom_coords = {}
        n_residues = pose.total_residue()
        
        # Initialize tensors for each atom type
        for res_idx in range(1, n_residues + 1):
            residue = pose.residue(res_idx)
            for atom_idx in range(1, residue.natoms() + 1):
                atom_name = residue.atom_name(atom_idx).strip()
                if atom_name not in atom_coords:
                    atom_coords[atom_name] = torch.zeros((n_residues, 3))
                
                # Get coordinates and convert to torch tensor
                xyz = pose.xyz(AtomID(atom_idx, res_idx))
                atom_coords[atom_name][res_idx - 1] = torch.tensor([xyz.x, xyz.y, xyz.z])
        
        return atom_coords 