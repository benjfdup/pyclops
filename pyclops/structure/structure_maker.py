# fundamentally, we want a class which will take in a chemical loss and a structure, of some kind, 
# and return a version of that structure with the chemical loss applied, which we can then minimize.

# this class would probably have a way to index what it needs to do for any given chemical loss, and
# then applies the appropriate steps to the structure, if that makes sense.
from typing import List
import MDAnalysis as mda
from MDAnalysis.core.universe import Merge
from rdkit import Chem

from ..core.chemical_loss import ChemicalLoss
from ..losses.disulfide import Disulfide
from ..losses.amide_losses import (AmideHead2Tail, 
                                   AmideSide2Side, 
                                   AmideSide2Head, 
                                   AmideSide2Tail,)
from ..losses.carboxylic_carbo import CarboxylicCarbo
from ..losses.cysteine_carbo import CysCarboxyl
from ..losses.lys_arg import LysArg
from ..losses.lys_tyr import LysTyr
from ..losses.standard_file_locations import STANDARD_LINKAGE_PDB_LOCATIONS
from ..utils.constants import AMBER_CAPS


class StructureMaker(): # relies jointly on MDAnalysis and RDKit
    """
    Class to handle topology manipulations suggested by PyCLOPS.
    Unfortunately, for now, this is only designed to work with cannonical 
    amino acids (not withstanding amber caps).
    """
    def __init__(self, 
                 initial_structure: mda.Universe,
                 ):
        self._initial_structure = self._clean_universe(initial_structure)

    @property
    def initial_structure(self) -> mda.Universe:
        return self._initial_structure

    @staticmethod
    def _clean_universe(universe: mda.Universe) -> mda.Universe:
        """
        Selects just protein atoms from a universe. Raises an error if there are no protein atoms,
        or if there are multiple protein chains. Returns a new Universe containing only that chain.
        """
        protein_atoms = universe.select_atoms("protein")
        if len(protein_atoms) == 0:
            raise ValueError("No protein atoms found in the universe")
        if len(protein_atoms.residues.segments) > 1:
            raise ValueError("Multiple protein chains found in the universe")
        
        # Create a new Universe from the protein atoms
        return Merge(protein_atoms)

    @classmethod
    def from_pdb(cls, 
                 pdb_filepath: str,
                 ) -> 'StructureMaker':
        """
        Creates a StructureMaker from a PDB file.
        """
        initial_structure = mda.Universe(pdb_filepath)
        return cls(initial_structure)
    
    @staticmethod
    def _remove_amber_caps( # TODO: confirm that this works.
        initial_structure: mda.Universe,
        remove_cap_str: str,
        verbose: bool = False,
    ):
        """
        Removes the amber caps from either the head, tail, or both. Replaces them with the 
        appropriate standard terminal atoms.

        remove_cap_str: "head", "tail", or "both"
        """
        # Validate input
        remove_cap_str = remove_cap_str.lower().strip() # should be fine since strings are immutable
        if remove_cap_str not in ["head", "tail", "both"]:
            raise ValueError(f"remove_cap_str must be 'head', 'tail', or 'both', got '{remove_cap_str}'")
        
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # Get all residues
        residues = list(structure.residues)
        if not residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Track atoms to remove
        atoms_to_remove = []
        
        # Check if the first residue's name is in AMBER_CAPS, we need to remove it if head or both
        if remove_cap_str in ["head", "both"]:
            first_residue = residues[0]
            if first_residue.resname in AMBER_CAPS:
                if verbose:
                    print(f"Removing head cap: {first_residue.resname}")
                # Get all atoms in the first residue
                atoms_to_remove.extend(list(first_residue.atoms))
        
        # Check if the last residue's name is in AMBER_CAPS, we need to remove it if tail or both
        if remove_cap_str in ["tail", "both"]:
            last_residue = residues[-1]
            # Only remove if it's different from the first residue (to avoid double removal)
            if last_residue != residues[0] and last_residue.resname in AMBER_CAPS:
                if verbose:
                    print(f"Removing tail cap: {last_residue.resname}")
                # Get all atoms in the last residue
                atoms_to_remove.extend(list(last_residue.atoms))
        
        # Remove atoms in reverse order to maintain indices
        for atom in sorted(atoms_to_remove, key=lambda x: x.index, reverse=True):
            structure.atoms.pop(atom.index)
        
        if verbose and atoms_to_remove:
            print(f"Removed {len(atoms_to_remove)} atoms from amber caps")
        
        return structure
    
    @staticmethod
    def _universe_to_mol(universe: mda.Universe, sanitize: bool = True) -> Chem.Mol:
        """
        Converts a universe to an RDKit molecule.
        """
        rdkit_mol = universe.atoms.convert_to("RDKIT")
        if sanitize:
            rdkit_mol = Chem.SanitizeMol(rdkit_mol)
        return rdkit_mol
    
    @staticmethod
    def save_universe_as_pdb(universe: mda.Universe,
                             filepath: str,
                             ) -> None:
        """
        Saves a universe as a PDB file.
        """
        universe.atoms.write(filepath)
        
    def make_molecule(self, 
                      chem_loss: ChemicalLoss,
                      ) -> mda.Universe:
        """
        Makes a molecule from an initial molecule and a chemical loss.

        The chemical loss instructs the molecule to undergo a specific cyclization chemistry
        between two amino acids.
        """
        # Amide bonds. Need to be careful about Amber caps.
        if isinstance(chem_loss, AmideHead2Tail):
            rem_cap_universe = self._remove_amber_caps(self.initial_structure, "both")
            rem_cap_mol = self._universe_to_mol(rem_cap_universe)

        else:
            raise ValueError(f"Chemical loss of type {type(chem_loss)} not supported.")