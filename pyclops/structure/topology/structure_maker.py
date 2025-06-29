"""
This module contains the StructureMaker class, 
which is used to make a structure given an initial structure and a ChemicalLoss.
"""
__all__ = [
    "StructureMaker",
]

from typing import Union, Dict, Tuple, List, Optional

import numpy as np
import torch
import MDAnalysis as mda
from rdkit import Chem

from ...core.chemical_loss.chemical_loss import ChemicalLoss

# Type aliases
ArrayLike = Union[torch.Tensor, np.ndarray]


class StructureMaker():
    """
    This class is used to make a structure given an initial 
    structure and a ChemicalLoss
    """

    def __init__(
            self,
            initial_universe: mda.Universe,
            ):
        self._validate_universe(initial_universe)
        self._initial_universe = initial_universe

    def _validate_universe(self,
                           universe: mda.Universe,
                           ) -> None:
        """
        Validate the universe to ensure it meets requirements for chemical modification.
        
        Checks:
        1. No water molecules are present
        2. Only one chain/segment is present  
        3. All atoms belong to protein residues
        
        Raises:
            ValueError: If any validation check fails
        """
        # Check 1: No water molecules
        common_water_names = ['HOH', 'TIP3', 'TIP4', 'WAT', 'SOL', 'H2O', 'SPC']
        water_selection = ' or '.join([f'resname {name}' for name in common_water_names])
        water_atoms = universe.select_atoms(water_selection)
        
        if len(water_atoms) > 0:
            unique_water_resnames = set(water_atoms.residues.resnames)
            raise ValueError(f"Water molecules detected: {unique_water_resnames}. "
                           "Please remove all water molecules from the structure.")
        
        # Check 2: Only one chain/segment
        n_segments = len(universe.segments)
        if n_segments != 1:
            raise ValueError(f"Structure must contain exactly one chain/segment. "
                           f"Found {n_segments} segments: {[seg.segid for seg in universe.segments]}")
        
        # Additional check for chain IDs if available
        if hasattr(universe.atoms, 'chainids'):
            unique_chains = set(universe.atoms.chainids)
            if len(unique_chains) > 1:
                raise ValueError(f"Structure must contain exactly one chain. "
                               f"Found {len(unique_chains)} chains: {sorted(unique_chains)}")
        
        # Check 3: All atoms must be protein atoms
        protein_atoms = universe.select_atoms("protein")
        total_atoms = len(universe.atoms)
        protein_atom_count = len(protein_atoms)
        
        if protein_atom_count == 0:
            raise ValueError("No protein atoms found in the structure. "
                           "Structure must contain protein residues.")
        
        if protein_atom_count != total_atoms:
            non_protein_atoms = total_atoms - protein_atom_count
            # Get examples of non-protein residues
            non_protein_residues = universe.select_atoms("not protein").residues
            unique_non_protein_resnames = set(non_protein_residues.resnames)
            
            raise ValueError(f"All atoms must belong to protein residues. "
                           f"Found {non_protein_atoms} non-protein atoms in residues: "
                           f"{sorted(unique_non_protein_resnames)}. "
                           f"Please ensure structure contains only protein atoms.")
        
        # Optional: Check for common issues
        n_residues = len(universe.residues)
        if n_residues == 0:
            raise ValueError("Structure must contain at least one residue.")

    @classmethod
    def from_pdb_file(cls, 
                      pdb_file: str,
                      ) -> 'StructureMaker':
        """
        Create a StructureMaker from a PDB file.
        """
        universe = mda.Universe(pdb_file)
        return cls(universe)

    def _set_positions(self, 
                       positions: ArrayLike, # shape: [n_atoms, 3]
                       units_factor: float,
                       ) -> None:
        """
        Set the positions of the atoms in the universe.
        units_factor is the factor by which the positions are scaled 
        (e.g. positions * units_factor = positions_in_Angstroms)

        Args:
            positions: ArrayLike, shape: [n_atoms, 3]
            units_factor: float, the factor by which the positions are scaled 
                (e.g. positions * units_factor = positions_in_Angstroms)
        """
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        
        # MDAnalysis expects positions in Angstroms
        self._initial_universe.atoms.positions = positions * units_factor

    def _initial_universe_to_rdkit_mol(self,
                                       ) -> Chem.Mol:
        """
        Convert the MDAnalysis universe to an RDKit molecule.

        Returns:
            Chem.Mol, the RDKit molecule
        """
        # Use MDAnalysis's built-in RDKit conversion
        ag = self._initial_universe.select_atoms("protein")
        return ag.to_rdkit()

    def _make_structure(self, 
                        chemical_loss: ChemicalLoss,
                        ) -> Chem.Mol:
        """
        Make a structure given a ChemicalLoss.
        """
        pass