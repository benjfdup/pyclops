from .base_scorer import BaseScorer
import torchmd as tm
import mdtraj as md
from typing import Optional, Union
import torch
import numpy as np
import tempfile
import os

TensorLike = Union[torch.Tensor, np.ndarray]


class TorchMDScorer(BaseScorer):
    def __init__(self,
                 topology: md.Topology,
                 units_factor: float,
                 forcefield: str = 'amber14-all.xml',
                 implicit_solvent_xml: str = 'implicit/gbn2.xml',
                 ):
        super().__init__(topology, units_factor)
        
        # Convert mdtraj topology to TorchMD format
        # Save topology to a temporary PDB to load with TorchMD
        
        # Create a temporary trajectory with the topology
        temp_traj = md.Trajectory(
            xyz=np.zeros((1, topology.n_atoms, 3)),
            topology=topology
        )
        
        # Save to temporary PDB file
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
            temp_pdb_path = tmp_file.name
            temp_traj.save_pdb(temp_pdb_path)
        
        try:
            # Load PDB with TorchMD
            self.pdb = tm.PDBFile(temp_pdb_path)
            
            # Set up forcefield based on solvent model
            self.forcefield = tm.ForceField(forcefield, implicit_solvent_xml)
            
            # Create system
            self.system = self.forcefield.createSystem(self.pdb.topology)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_pdb_path)

    @classmethod
    def from_pdb_file(cls,
                      pdb_file: str,
                      units_factor: float,
                      forcefield: str = 'amber14-all.xml',
                      implicit_solvent_xml: str = 'implicit/gbn2.xml',
                      ) -> 'TorchMDScorer':
        """
        Create a TorchMDScorer from a PDB file.
        """
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor, forcefield, implicit_solvent_xml)

    def _prepare_coordinates(self, coordinates: TensorLike) -> torch.Tensor:
        """Convert coordinates to TorchMD format and handle batching"""
        # Convert to torch if numpy array
        if isinstance(coordinates, np.ndarray):
            coords_torch = torch.from_numpy(coordinates).float()
        else:
            coords_torch = coordinates.clone()
        
        # Convert to angstroms
        coords_angstrom = self._convert_to_angstroms(coords_torch)
        
        return coords_angstrom

    def calculate_energy(self,
                         coordinates: TensorLike,
                         ) -> TensorLike:
        """
        Calculate potential energy in kJ/mol.

        Args:
            coordinates: TensorLike, shape: [n_batch, n_atoms, 3]

        Returns:
            TensorLike, shape: [n_batch]
        """
        coords_angstrom = self._prepare_coordinates(coordinates)
        is_torch = isinstance(coordinates, torch.Tensor)
        
        # Handle batching
        if coords_angstrom.ndim == 3:  # [n_batch, n_atoms, 3]
            batch_size = coords_angstrom.shape[0]
            energies = []
            
            for i in range(batch_size):
                # Set positions and calculate energy
                self.system.setPositions(coords_angstrom[i])
                energy_kj = self.system.getPotentialEnergy().value_in_unit(tm.unit.kilojoules_per_mole)
                energies.append(energy_kj)
            
            result = torch.tensor(energies, dtype=torch.float32)  # Shape: [n_batch]
        else:  # Single conformation [n_atoms, 3]
            self.system.setPositions(coords_angstrom)
            energy_kj = self.system.getPotentialEnergy().value_in_unit(tm.unit.kilojoules_per_mole)
            result = torch.tensor([energy_kj], dtype=torch.float32)  # Shape: [1]
        
        # Convert back to numpy if input was numpy
        if not is_torch:
            return result.numpy()
        
        # Move to same device as input if torch
        if isinstance(coordinates, torch.Tensor):
            result = result.to(coordinates.device)
        
        return result