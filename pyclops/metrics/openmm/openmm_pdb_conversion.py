"""
Conversion utilities between PDB + atom coordinates and OpenMM objects.
"""

from typing import Tuple, Dict, Optional, Union
import torch
import numpy as np
import tempfile
import os
from openmm import app, unit
from openmm.app import PDBFile

class OpenMMConverter:
    def __init__(self):
        """Initialize the converter."""
        pass
    
    def pdb_to_coordinates(self, pdb_path: str) -> Tuple[app.Topology, np.ndarray]:
        """
        Convert a PDB file to OpenMM topology and coordinates.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            Tuple containing:
                - OpenMM Topology object
                - Array of coordinates (shape: n_atoms × 3)
        """
        pdb = PDBFile(pdb_path)
        positions = pdb.positions.value_in_unit(unit.nanometers)
        return pdb.topology, positions
    
    def coordinates_to_pdb(self,
                          pdb_path: str,
                          new_coords: Union[torch.Tensor, np.ndarray],
                          output_path: Optional[str] = None) -> str:
        """
        Create a new PDB file with updated coordinates.
        
        Args:
            pdb_path: Path to the template PDB file
            new_coords: Tensor/array of shape (n_atoms, 3) containing new coordinates
            output_path: Optional path to save the new PDB. If None, creates a temporary file.
            
        Returns:
            Path to the new PDB file
        """
        # Load template PDB
        pdb = PDBFile(pdb_path)
        
        # Convert coordinates to numpy if they're torch tensors
        if isinstance(new_coords, torch.Tensor):
            new_coords = new_coords.cpu().numpy()
            
        # Convert to OpenMM units
        positions = new_coords * unit.nanometers
        
        # Create new PDB file
        if output_path is None:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_file:
                output_path = temp_file.name
        
        # Write new PDB
        with open(output_path, 'w') as f:
            PDBFile.writeFile(pdb.topology, positions, f)
            
        return output_path
    
    def extract_atom_info(self, pdb_path: str) -> Dict:
        """
        Extract atom information from a PDB file.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            Dictionary containing atom information:
                - 'residues': List of residue names
                - 'atoms': List of atom names
                - 'chains': List of chain IDs
        """
        pdb = PDBFile(pdb_path)
        
        info = {
            'residues': [],
            'atoms': [],
            'chains': []
        }
        
        for chain in pdb.topology.chains():
            for residue in chain.residues():
                for atom in residue.atoms():
                    info['residues'].append(residue.name)
                    info['atoms'].append(atom.name)
                    info['chains'].append(chain.id)
        
        return info 