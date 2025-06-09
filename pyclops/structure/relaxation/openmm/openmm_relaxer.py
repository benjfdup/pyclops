import torch
import numpy as np
from typing import Union, Optional
import openmm as mm
import openmm.app as app
from openmm import unit
from ..base_relaxer import BaseRelaxer


class OpenMMRelaxer(BaseRelaxer):
    def __init__(self, 
                 units_factor: float, 
                 pdb_filepath: str,
                 forcefield: Optional[str] = 'amber14-all.xml',
                 water_model: Optional[str] = None,
                 max_iterations: int = 1000,
                 tolerance: float = 10.0):
        """
        Args:
            units_factor: factor to convert input coordinates to angstroms
            pdb_filepath: path to PDB file
            forcefield: OpenMM forcefield XML file
            water_model: OpenMM water model XML file (None for implicit solvent)
            max_iterations: maximum number of minimization iterations
            tolerance: energy tolerance for minimization (kJ/mol/nm)
        """
        super().__init__(units_factor, pdb_filepath)
        
        # Store minimization parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Load PDB and create OpenMM system
        pdb = app.PDBFile(pdb_filepath)
        self.topology = pdb.topology
        
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
        
        # Create integrator (required for context, but not used in minimization)
        integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
        
        # Create simulation context
        self.context = mm.Context(self.system, integrator)

    def _convert_to_angstroms(self, coordinates: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Convert input coordinates to angstroms"""
        return coordinates * self.units_factor
    
    def _convert_from_angstroms(self, coordinates: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Convert coordinates from angstroms to original units"""
        return coordinates / self._units_factor

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

    def _minimize_single_structure(self, coordinates_nm: np.ndarray) -> np.ndarray:
        """Minimize a single structure and return relaxed coordinates in nm"""
        # Set positions
        self.context.setPositions(coordinates_nm * unit.nanometer)
        
        # Perform energy minimization
        mm.LocalEnergyMinimizer.minimize(
            self.context,
            tolerance=self.tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=self.max_iterations
        )
        
        # Get minimized positions
        state = self.context.getState(getPositions=True)
        minimized_positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        
        return minimized_positions

    def relax(self, structure: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Relax molecular structure using OpenMM energy minimization.
        
        Args:
            structure: Input coordinates. Can be:
                - Single structure: [n_atoms, 3] 
                - Batch of structures: [n_batch, n_atoms, 3]
        
        Returns:
            Relaxed coordinates in the same format and units as input
        """
        coords_nm = self._prepare_coordinates(structure)
        is_torch = isinstance(structure, torch.Tensor)
        
        # Handle batching
        if coords_nm.ndim == 3:  # [n_batch, n_atoms, 3]
            batch_size = coords_nm.shape[0]
            relaxed_coords = []
            
            for i in range(batch_size):
                minimized = self._minimize_single_structure(coords_nm[i])
                relaxed_coords.append(minimized)
            
            result_nm = np.array(relaxed_coords)
        else:  # Single conformation [n_atoms, 3]
            minimized = self._minimize_single_structure(coords_nm)
            result_nm = np.array([minimized])
        
        # Convert back to original units (angstroms, then to input units)
        result_angstrom = result_nm * 10.0  # nm to angstrom
        result_original_units = self._convert_from_angstroms(result_angstrom)
        
        # Remove batch dimension if input was single structure
        if structure.ndim == 2:
            result_original_units = result_original_units[0]
        
        # Convert back to torch if input was torch
        if is_torch:
            return torch.from_numpy(result_original_units).float().to(structure.device)
        
        return result_original_units