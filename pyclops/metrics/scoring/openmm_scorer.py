import mdtraj as md

from typing import Union, Optional
import torch
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

import tempfile
import os

from .base_scorer import BaseScorer

# Type aliases
TensorLike = Union[torch.Tensor, np.ndarray]

class OpenMMScorer(BaseScorer):
    def __init__(self,
                 topology: md.Topology,
                 units_factor: float,
                 forcefield: str = 'amber14-all.xml',
                 implicit_solvent_xml: str = 'implicit/gbn2.xml',
                 ):
        """
        Initialize OpenMM scorer.
        
        Args:
            topology: MDTraj topology
            units_factor: Factor to convert input coordinates to angstroms
            forcefield: OpenMM forcefield XML file
            implicit_solvent_xml: Implicit solvent XML file
        """
        super().__init__(topology, units_factor)
        
        # Convert mdtraj topology to OpenMM topology
        # Save topology to a temporary PDB to load with OpenMM
        
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
            # Load PDB with OpenMM
            pdb = app.PDBFile(temp_pdb_path)
            
            # Set up forcefield based on solvent model
            # Implicit solvent - include the implicit solvent XML file
            forcefield_obj = app.ForceField(forcefield, implicit_solvent_xml)
            self.system = forcefield_obj.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                # No need to pass implicitSolvent parameter - it's defined in the XML
                )
            
            # Create integrator (dummy, we only need it for context)
            integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
            
            # Create simulation context
            self.context = mm.Context(self.system, integrator)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_pdb_path)

    @classmethod
    def from_pdb_file(cls,
                      pdb_file: str,
                      units_factor: float,
                      forcefield: str = 'amber14-all.xml',
                      implicit_solvent_xml: str = 'implicit/obc2.xml',
                      ) -> 'OpenMMScorer':
        """
        Create an OpenMMScorer from a PDB file.
        """
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor, forcefield, implicit_solvent_xml)

    def _convert_to_angstroms(self, coordinates: np.ndarray) -> np.ndarray:
        """Convert input coordinates to angstroms using units_factor"""
        return coordinates * self.units_factor

    def _prepare_coordinates(self, coordinates: TensorLike) -> np.ndarray:
        """Convert coordinates to OpenMM format (nm) and handle batching"""
        # Convert to numpy if torch tensor
        if isinstance(coordinates, torch.Tensor):
            coords_np = coordinates.detach().cpu().numpy()
        else:
            coords_np = coordinates.copy()
        
        # Convert to angstroms, then to nanometers for OpenMM
        coords_angstrom = self._convert_to_angstroms(coords_np)
        coords_nm = coords_angstrom / 10.0  # angstrom to nanometer
        
        return coords_nm

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
        coords_nm = self._prepare_coordinates(coordinates)
        is_torch = isinstance(coordinates, torch.Tensor)
        
        # Handle batching
        if coords_nm.ndim == 3:  # [n_batch, n_atoms, 3]
            batch_size = coords_nm.shape[0]
            energies = []
            
            for i in range(batch_size):
                # Set positions and calculate energy
                self.context.setPositions(coords_nm[i] * unit.nanometer)
                state = self.context.getState(getEnergy=True)
                energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                energies.append(energy_kj)
            
            result = np.array(energies)  # Shape: [n_batch]
        else:  # Single conformation [n_atoms, 3]
            self.context.setPositions(coords_nm * unit.nanometer)
            state = self.context.getState(getEnergy=True)
            energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            result = np.array([energy_kj])  # Shape: [1]
        
        # Convert back to torch if input was torch
        if is_torch:
            return torch.from_numpy(result).float().to(coordinates.device)
        return result