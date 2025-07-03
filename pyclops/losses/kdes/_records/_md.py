################################################################################################
# This code is essentially adapted (taken) from the blog of Corin Wagen
# I have made some modifications to it, but you should cite the following blog post
# if you use it. Thanks!
# https://corinwagen.github.io/public/blog/20240613_simple_md.html
################################################################################################

import os
import argparse
import yaml
from typing import Callable

from openff.toolkit import Molecule, Topology

from openmm import *
from openmm.app import *

import numpy as np
import openmoltools
import tempfile
import cctk
import tqdm

from sys import stdout

from rdkit import Chem
from rdkit.Chem import AllChem

from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from openmm.app.element import hydrogen, oxygen

import mdtraj as md

from loguru import logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run molecular dynamics simulation')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to YAML configuration file')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Parse command line arguments and load configuration
args = parse_args()
config = load_config(args.config)

# Extract parameters from config
smiles = config['smiles']
save_dir = config['save_dir']
seeds = range(config['seeds'])
checkpoint_interval = config['checkpoint_interval']
printout_interval = config['printout_interval']
steps = config['steps']

os.makedirs(save_dir, exist_ok=True)

# Configure loguru to log to both console and file
log_file = os.path.join(save_dir, "md_simulation.log")
logger.remove()  # Remove default handler
logger.add(stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

logger.info('----------========== Starting MD ==========----------')

def generate_forcefield(smiles: str) -> ForceField:
    """ Creates an OpenMM ForceField object that knows how to handle a given SMILES string """
    molecule = Molecule.from_smiles(smiles)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield = ForceField(
      'amber/protein.ff14SB.xml',
      'amber/tip3p_standard.xml',
      'amber/tip3p_HFE_multivalent.xml'
     )
    forcefield.registerTemplateGenerator(smirnoff.generator)
    return forcefield

def generate_initial_pdb(
    smiles: str,
    perm_save_dir: str, # directory where to permanently save the system.pdb file...
    min_side_length: int = 25, # Å
    solvent_smiles = "O",
) -> PDBFile:
    """ Creates a PDB file for a solvated molecule, starting from two SMILES strings. """

    # do some math to figure how big the box needs to be
    solute = cctk.Molecule.new_from_smiles(smiles)
    solute_volume = solute.volume(qhull=True)
    solvent = cctk.Molecule.new_from_smiles(solvent_smiles)
    solvent_volume = solvent.volume(qhull=False)

    total_volume = 50 * solute_volume # seems safe?
    min_allowed_volume = min_side_length ** 3
    total_volume = max(min_allowed_volume, total_volume)

    total_solvent_volume = total_volume - solute_volume
    n_solvent = int(total_solvent_volume // solvent_volume)
    box_size = total_volume ** (1/3)

    # build pdb
    with tempfile.TemporaryDirectory() as tempdir:
        solute_fname = f"{tempdir}/solute.pdb"
        solvent_fname = f"{tempdir}/solvent.pdb"
        system_fname = f"system.pdb"

        smiles_to_pdb(smiles, solute_fname)
        smiles_to_pdb(solvent_smiles, solvent_fname)
        traj_packmol = openmoltools.packmol.pack_box(
          [solute_fname, solvent_fname],
          [1, n_solvent],
          box_size=box_size
         )
        traj_packmol.save_pdb(system_fname)
        traj_packmol.save_pdb(os.path.join(perm_save_dir, 'system.pdb')) # saves to permanent directory.

        return PDBFile(system_fname)

def smiles_to_pdb(smiles: str, filename: str) -> None:
    """ Turns a SMILES string into a PDB file (written to current working directory). """
    m = Chem.MolFromSmiles(smiles)
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh)
    Chem.MolToPDBFile(mh, filename)

class SubsettingDCDReporter(object): # Taken from OpenMM Github (doesnt seem to be on the conda package)
    """DCDReporter outputs a series of frames from a Simulation to a DCD file.

    To use it, create a DCDReporter, then add it to the Simulation's list of reporters.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None, atomSubset=None):
        """Create a DCDReporter.

        Parameters
        ----------
        file : string
            The file to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        append : bool=False
            If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: bool
            Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
        atomSubset: list
            Atom indices (zero indexed) of the particles to output.  If None (the default), all particles will be output.
        """
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        self._atomSubset = atomSubset
        if append:
            mode = 'r+b'
        else:
            mode = 'wb'
        self._out = open(file, mode)
        self._dcd = None

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        dict
            A dictionary describing the required information for the next report
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return {'steps':steps, 'periodic':self._enforcePeriodicBox, 'include':['positions']}

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        if self._dcd is None:
            if self._atomSubset is None:
                topology = simulation.topology
            else:
                topology = Topology()
                topology.setPeriodicBoxVectors(simulation.topology.getPeriodicBoxVectors())
                atoms = list(simulation.topology.atoms())
                chain = topology.addChain()
                residue = topology.addResidue('', chain)
                for i in self._atomSubset:
                    topology.addAtom(atoms[i].name, atoms[i].element, residue)
            self._dcd = DCDFile(
                self._out, topology, simulation.integrator.getStepSize(),
                simulation.currentStep, self._reportInterval, self._append
            )
        positions = state.getPositions(asNumpy=True)
        if self._atomSubset is not None:
            positions = [positions[i] for i in self._atomSubset]
        self._dcd.writeModel(positions, periodicBoxVectors=state.getPeriodicBoxVectors())

    def __del__(self):
        self._out.close()

logger.info(f"Generating forcefield for SMILES: {smiles}")
forcefield = generate_forcefield(smiles)
logger.info(f"Generating initial PDB structure and saving to {save_dir}")
pdb = generate_initial_pdb(smiles, perm_save_dir = save_dir, solvent_smiles="O")

logger.info("Creating OpenMM system with PME and 1nm cutoff")
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1*unit.nanometer,
)

def get_non_water_indices(topology) -> np.ndarray:
    """
    Extracts indices of non-water atoms from the given OpenMM topology.
    A water molecule is defined as a chain with ≤3 atoms, all of which are H or O.
    """
    non_water_indices = []

    for chain in topology.chains():
        atoms = list(chain.atoms())
        atom_elements = [atom.element for atom in atoms]

        # Check if all atoms are hydrogen or oxygen, and total atom count ≤ 3
        if len(atoms) <= 3 and all(e in (hydrogen, oxygen) for e in atom_elements):
            continue  # likely water → skip
        else:
            non_water_indices.extend(atom.index for atom in atoms)

    return np.array(non_water_indices, dtype=int)

def save_non_water_pdb(initialPdb: str, out_path: str) -> None:
    """
    Save a PDB file excluding chains that are likely water molecules.

    A water chain is defined as one with ≤3 atoms, all of which are hydrogen or oxygen.
    All other chains are preserved with full connectivity and ordering.

    Parameters:
        initialPdb (str): Path to input PDB file (with topology and positions).
        out_path (str): Path to save the filtered PDB.
    """
    # Load the trajectory (just the first frame is fine for structure)
    traj = md.load(initialPdb)

    topology = traj.topology
    atoms_to_keep = []

    for chain in topology.chains:
        chain_atoms = list(chain.atoms)
        if len(chain_atoms) <= 3:
            # Check if all atoms are H or O
            if all(atom.element.symbol in ('H', 'O') for atom in chain_atoms):
                continue  # Skip this water-like chain
        atoms_to_keep.extend(chain_atoms)

    # Get indices of atoms to keep
    atom_indices = [atom.index for atom in atoms_to_keep]

    # Slice the trajectory
    filtered_traj = traj.atom_slice(atom_indices)

    # Save to file
    filtered_traj.save_pdb(out_path)
logger.info("Identifying non-water atoms and creating filtered PDB")
non_water_idxs = get_non_water_indices(pdb.topology)
save_non_water_pdb(initialPdb = os.path.join(save_dir, "system.pdb"), 
                   out_path = os.path.join(save_dir, "filtered_system.pdb"))
logger.info(f"Found {len(non_water_idxs)} non-water atoms out of {pdb.topology.getNumAtoms()} total atoms")

file_exists: Callable[[str], bool] = lambda filepath: os.path.exists(filepath)

logger.info(f"Starting MD simulations for {len(seeds)} seeds with {steps} steps each")
for seed in tqdm.tqdm(seeds):
    logger.info(f"Starting simulation for seed {seed}")
    traj_file = os.path.join(save_dir, f"traj_seed_{seed}.dcd")
    no_water_traj_file = os.path.join(save_dir, f"traj_seed_{seed}_no_water.dcd")
    csv_file = os.path.join(save_dir, f"scalars_seed_{seed}.csv")

    # initialize Langevin integrator and minimize
    integrator = LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds)
    integrator.setRandomNumberSeed(seed)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    # we'll make this an NPT simulation now
    system.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
    simulation.context.reinitialize(preserveState=True)

    # set the reporters collecting the MD output.
    simulation.reporters = []

    simulation.reporters.append(DCDReporter(traj_file, checkpoint_interval)) # replace this with a subsetting one
    simulation.reporters.append(
        SubsettingDCDReporter(
            file = no_water_traj_file,
            reportInterval= checkpoint_interval,
            atomSubset= non_water_idxs
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            printout_interval,
            step=True,
            temperature=True,
            elapsedTime=True,
            volume=True,
            density=True
        )
    )

    simulation.reporters.append(
        StateDataReporter(
            csv_file,
            checkpoint_interval,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        )
    )

    # actually run the MD
    logger.info(f"Running {steps} MD steps for seed {seed}")
    simulation.step(steps) # this is the number of steps, you may want fewer to test quickly
    logger.info(f"Completed simulation for seed {seed}")

logger.info('----------========== MD Simulations Complete ==========----------')
logger.info(f"All results saved to: {save_dir}")