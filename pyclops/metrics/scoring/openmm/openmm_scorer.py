from typing import Union, Optional
import torch
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

from ..base_scorer import BaseScorer


class OpenMMScorer(BaseScorer):
    def __init__(self, 
                 pdb_path: str, 
                 units_factor: float,
                 forcefield: str = 'amber14-all.xml',
                 water_model: Optional[str] = 'amber14/tip3pfb.xml'):
        """
        Args:
            pdb_path: path to PDB file
            units_factor: factor to convert input coordinates to angstroms
            forcefield: OpenMM forcefield XML file
            water_model: OpenMM water model XML file (None for implicit solvent)
        """
        super().__init__(pdb_path, units_factor)
        
        # Load PDB and create OpenMM system
        pdb = app.PDBFile(pdb_path)
        
        # Set up forcefield
        if water_model:
            forcefield_obj = app.ForceField(forcefield, water_model)
        else:
            forcefield_obj = app.ForceField(forcefield)
        
        # Create system
        self.system = forcefield_obj.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME if water_model else app.NoCutoff,
            nonbondedCutoff=1*unit.nanometer if water_model else None,
        )
        
        # Create integrator (dummy, we only need it for context)
        integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
        
        # Create simulation context
        self.context = mm.Context(self.system, integrator)
        
        # Unit conversion factors
        self.force_conversion = 10.0  # kJ/mol/nm to kJ/mol/angstrom

    def _prepare_coordinates(self, coordinates: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
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
                         coordinates: Union[torch.Tensor, np.ndarray],
                         ) -> Union[torch.Tensor, np.ndarray]:
        """Calculate potential energy in kJ/mol"""
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

    def calculate_forces(self, 
                         coordinates: Union[torch.Tensor, np.ndarray],
                         ) -> Union[torch.Tensor, np.ndarray]:
        """Calculate forces in kJ/mol/angstrom"""
        coords_nm = self._prepare_coordinates(coordinates)
        is_torch = isinstance(coordinates, torch.Tensor)
        
        # Handle batching
        if coords_nm.ndim == 3:  # [n_batch, n_atoms, 3]
            batch_size = coords_nm.shape[0]
            forces_list = []
            
            for i in range(batch_size):
                # Set positions and calculate forces
                self.context.setPositions(coords_nm[i] * unit.nanometer)
                state = self.context.getState(getForces=True)
                forces_kj_nm = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
                forces_kj_ang = forces_kj_nm * self.force_conversion
                forces_list.append(forces_kj_ang)
            
            result = np.array(forces_list)
        else:  # Single conformation [n_atoms, 3]
            self.context.setPositions(coords_nm * unit.nanometer)
            state = self.context.getState(getForces=True)
            forces_kj_nm = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
            result = np.array([forces_kj_nm * self.force_conversion])
        
        # Convert back to torch if input was torch
        if is_torch:
            return torch.from_numpy(result).float().to(coordinates.device)
        return result