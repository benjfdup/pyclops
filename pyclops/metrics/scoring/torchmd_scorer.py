from .base_scorer import BaseScorer
import torchmd as tm
import mdtraj as md
from typing import Union, Optional
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
                 device: Optional[str] = None,
                 ):
        super().__init__(topology, units_factor)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        
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
            pdb = tm.PDBFile(temp_pdb_path)
            
            # Create TorchMD system with forcefield
            ff = tm.ForceField(forcefield, implicit_solvent_xml)
            
            # Create the system
            system = ff.createSystem(pdb.topology)
            
            # Create simulation configuration
            self.conf = {
                'system': system,
                'topology': pdb.topology,
                'integrator': tm.integrators.VelocityVerlet(1.0),  # dummy timestep
                'device': torch.device(device),
                'precision': torch.float32,
            }
            
            # Create the simulation object
            self.sim = tm.dynamics.NVT(**self.conf)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_pdb_path)

    @classmethod
    def from_pdb_file(cls,
                      pdb_file: str,
                      units_factor: float,
                      forcefield: str = 'amber14-all.xml',
                      implicit_solvent_xml: str = 'implicit/gbn2.xml',
                      device: str = 'cpu',
                      ) -> 'TorchMDScorer':
        """
        Create a TorchMDScorer from a PDB file.
        """
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor, forcefield, implicit_solvent_xml, device)

    def _prepare_coordinates(self, coordinates: TensorLike) -> torch.Tensor:
        """Convert coordinates to TorchMD format and handle batching"""
        # Convert to torch if numpy array
        if isinstance(coordinates, np.ndarray):
            coords_torch = torch.from_numpy(coordinates).float()
        else:
            coords_torch = coordinates.clone()
        
        # Convert to angstroms
        coords_angstrom = self._convert_to_angstroms(coords_torch)
        
        # Move to device
        coords_angstrom = coords_angstrom.to(self.device)
        
        return coords_angstrom

    def calculate_energy(self,
                         coordinates: TensorLike,
                         ) -> TensorLike:
        """
        Calculate potential energy in kJ/mol.

        Args:
            coordinates: TensorLike, shape: [n_batch, n_atoms, 3] or [n_atoms, 3]

        Returns:
            TensorLike, shape: [n_batch] or [1]
        """
        coords_angstrom = self._prepare_coordinates(coordinates)
        is_torch = isinstance(coordinates, torch.Tensor)
        original_device = coordinates.device if isinstance(coordinates, torch.Tensor) else None
        
        # Handle batching
        if coords_angstrom.ndim == 3:  # [n_batch, n_atoms, 3]
            batch_size = coords_angstrom.shape[0]
            energies = []
            
            for i in range(batch_size):
                # Set positions in TorchMD simulation
                self.sim.pos = coords_angstrom[i]
                
                # Calculate energy using TorchMD's force calculation
                # This will compute the potential energy
                energy = self.sim.calculate_potential_energy()
                
                # Convert from kcal/mol to kJ/mol (TorchMD uses kcal/mol)
                energy_kj = energy * 4.184
                energies.append(energy_kj.item())
            
            result = torch.tensor(energies, dtype=torch.float32)  # Shape: [n_batch]
        else:  # Single conformation [n_atoms, 3]
            # Set positions in TorchMD simulation
            self.sim.pos = coords_angstrom
            
            # Calculate energy
            energy = self.sim.calculate_potential_energy()
            
            # Convert from kcal/mol to kJ/mol
            energy_kj = energy * 4.184
            result = torch.tensor([energy_kj.item()], dtype=torch.float32)  # Shape: [1]
        
        # Convert back to numpy if input was numpy
        if not is_torch:
            return result.cpu().numpy()
        
        # Move to same device as input if torch
        if original_device is not None:
            result = result.to(original_device)
        
        return result