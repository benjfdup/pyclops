"""
All-atom scoring functions for biomolecules using OpenMM.
"""

from typing import Dict, Optional, Union, Tuple
import torch
import numpy as np
from openmm import app, unit
from openmm.app import PDBFile
from openmm import openmm

class OpenMMScorer:
    def __init__(self, 
                 forcefield: str = 'amber14-all.xml',
                 water_model: str = 'amber14/tip3p.xml',
                 implicit_solvent: Optional[str] = None):
        """
        Initialize the OpenMM scorer.
        
        Args:
            forcefield: Name of the forcefield XML file
            water_model: Name of the water model XML file
            implicit_solvent: Optional implicit solvent model ('GBSA' or 'OBC')
        """
        self.forcefield = forcefield
        self.water_model = water_model
        self.implicit_solvent = implicit_solvent
        
    def create_system(self, pdb_path: str) -> Tuple[app.Simulation, openmm.System]:
        """
        Create an OpenMM system from a PDB file.
        
        Args:
            pdb_path: Path to the PDB file
            
        Returns:
            Tuple of (Simulation, System) objects
        """
        # Load the PDB file
        pdb = PDBFile(pdb_path)
        
        # Create forcefield
        forcefield = app.ForceField(self.forcefield, self.water_model)
        
        # Create system
        if self.implicit_solvent:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                implicitSolvent=self.implicit_solvent
            )
        else:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff
            )
            
        # Create simulation
        integrator = openmm.LangevinIntegrator(
            300*unit.kelvin,
            1/unit.picosecond,
            2*unit.femtoseconds
        )
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        return simulation, system
    
    def calculate_energy(self, 
                        pdb_path: str,
                        coordinates: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Dict[str, float]:
        """
        Calculate the energy of a structure.
        
        Args:
            pdb_path: Path to the PDB file
            coordinates: Optional new coordinates to use (shape: n_atoms × 3)
            
        Returns:
            Dictionary of energy components
        """
        simulation, system = self.create_system(pdb_path)
        
        if coordinates is not None:
            # Convert coordinates to OpenMM units
            if isinstance(coordinates, torch.Tensor):
                coordinates = coordinates.cpu().numpy()
            positions = coordinates * unit.nanometers
            simulation.context.setPositions(positions)
        
        # Get state
        state = simulation.context.getState(getEnergy=True)
        
        # Extract energy components
        energy_components = {
            'total': state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            'kinetic': state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        }
        
        # Get force group energies
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if hasattr(force, 'getForceGroup'):
                group = force.getForceGroup()
                energy = simulation.context.getState(
                    getEnergy=True,
                    groups={group}
                ).getPotentialEnergy()
                energy_components[force.__class__.__name__] = energy.value_in_unit(unit.kilojoules_per_mole)
        
        return energy_components
    
    def calculate_forces(self,
                        pdb_path: str,
                        coordinates: Optional[Union[torch.Tensor, np.ndarray]] = None) -> np.ndarray:
        """
        Calculate forces on atoms.
        
        Args:
            pdb_path: Path to the PDB file
            coordinates: Optional new coordinates to use (shape: n_atoms × 3)
            
        Returns:
            Array of forces (shape: n_atoms × 3)
        """
        simulation, _ = self.create_system(pdb_path)
        
        if coordinates is not None:
            if isinstance(coordinates, torch.Tensor):
                coordinates = coordinates.cpu().numpy()
            positions = coordinates * unit.nanometers
            simulation.context.setPositions(positions)
        
        # Get forces
        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        
        # Convert to numpy array in kJ/mol/nm
        return forces.value_in_unit(unit.kilojoules_per_mole/unit.nanometer) 