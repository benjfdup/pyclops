"""
This module contains the StructureMaker class, 
which is used to make a structure given an initial structure and a ChemicalLoss.
"""
__all__ = [
    "StructureMaker",
]

from typing import Union, Dict, Type, Optional
import tempfile
import os

import numpy as np
import torch
import MDAnalysis as mda
from rdkit import Chem

from ...core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict, AtomKey
from ...core.loss_handler.chemical_loss_handler import ChemicalLossHandler
from .utils import DEFAULT_MODIFIER_DICT
from .loss_structure_modifier import LossStructureModifier
from .structure_modifier.amide_modifier import Amide2TermModifier
from .structure_modifier.carboxylic_carbo_modifier import CarboxylicCarbo2CTermModifier

# Type aliases
ArrayLike = Union[torch.Tensor, np.ndarray]


class StructureMaker():
    """
    This class is used to make a structure given an initial 
    structure (via a ChemicalLossHandler) and a ChemicalLoss.
    It effectively applies the ChemicalLoss to the initial structure
    in the form of a universe provided by the mdtraj of the 
    ChemicalLossHandler.
    """

    _chem_loss_to_modifier_type: Dict[str, Type[LossStructureModifier]] = DEFAULT_MODIFIER_DICT

    def __init__(
            self,
            chemical_loss_handler: ChemicalLossHandler,
            ):
        self._chemical_loss_handler = chemical_loss_handler
        self._initial_universe = self._chemical_loss_handler_to_universe()
        self._initial_rdkit_mol = self._initial_universe_to_rdkit_mol()
        self._mdtraj_atom_idxs_dict = self._chemical_loss_handler.atom_indexes_dict
        self._rdkit_atom_idxs_dict = self._parse_rdkit_mol(self._initial_rdkit_mol)

    
    def _chemical_loss_handler_to_universe(self) -> mda.Universe:
        """
        Convert the self._chemical_loss_handler ChemicalLossHandler to an MDAnalysis 
        universe. Uses the first frame of the MDTraj trajectory for initial positions.

        Returns:
            mda.Universe, the MDAnalysis universe
        """
        # Get the MDTraj trajectory from the chemical loss handler
        traj = self._chemical_loss_handler._traj
        
        # Create a temporary PDB file with the first frame
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
            # Save the first frame to PDB
            traj[0].save_pdb(tmp_file.name)
            
            # Load with MDAnalysis
            universe = mda.Universe(tmp_file.name)
            
            # Clean up the temporary file
            os.unlink(tmp_file.name)
        
        return universe

    def _set_positions(self, 
                       positions: ArrayLike, # shape: [n_atoms, 3]
                       units_factor: float,
                       ) -> None:
        """
        Set the positions of the atoms in the universe.
        units_factor is the factor by which the positions are scaled 
        (e.g. positions * units_factor = positions_in_Angstroms)

        Should be called calling the `_make_structure` method, as that removes atoms 
        etc, hence ruining the indexing.

        Args:
            positions: ArrayLike, shape: [n_atoms, 3]
            units_factor: float, the factor by which the positions are scaled 
                (e.g. positions * units_factor = positions_in_Angstroms)
        """
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        
        # MDAnalysis expects positions in Angstroms
        self._initial_universe.atoms.positions = positions * units_factor

    def _initial_universe_to_rdkit_mol(self) -> Chem.Mol:
        """
        Convert the MDAnalysis universe to an RDKit molecule.

        Returns:
            Chem.Mol, the RDKit molecule
        """
        # Create a temporary directory and file for the PDB
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = os.path.join(temp_dir, "temp_structure.pdb")
            
            # Save the universe to PDB file
            self._initial_universe.atoms.write(pdb_file)
            
            # Read the PDB file with RDKit
            mol = Chem.MolFromPDBFile(pdb_file, sanitize=True, removeHs=True)
            
            if mol is None:
                raise ValueError("Failed to create RDKit molecule from PDB file")
            
            return mol
        
    @staticmethod
    def _parse_rdkit_mol(
        mol: Chem.Mol,
        ) -> AtomIndexDict:
        """
        Parse the RDKit molecule.
        Returns a dictionary mapping from (residue index, residue name) to atom indexes.
        This is used to map the atom indexes to the residue index and atom name.
        """
        n_non_hydrogens = len(mol.GetAtoms())
        residue_idx_atom_name_to_atom_idx: AtomIndexDict = {}

        for atom_idx in range(n_non_hydrogens):
            atom = mol.GetAtomWithIdx(atom_idx)
            residue_idx: int = atom.GetPDBResidueInfo().GetResidueNumber() - 1 # RDKit is 1-indexed for residues, 
                                                                               # but we want 0-indexed for the residue index
            atom_name: str = atom.GetPDBResidueInfo().GetName().strip().upper()
            atom_key: AtomKey = (residue_idx, atom_name)
            residue_idx_atom_name_to_atom_idx[atom_key] = atom_idx

        return residue_idx_atom_name_to_atom_idx
    
    @property
    def initial_rdkit_mol(self) -> Chem.Mol:
        """
        A deep copy of the initial RDKit molecule.
        """
        return Chem.Mol(self._initial_rdkit_mol)
    
    def _is_terminal_subclass(self,
                              modifier_type: Type[LossStructureModifier],
                              ) -> bool:
        """
        Determine if the modifier is a subclass which connects to a terminal group.
        """
        terminal_subclasses = (Amide2TermModifier, 
                               CarboxylicCarbo2CTermModifier)
        return any(issubclass(modifier_type, cls) for cls in terminal_subclasses)
    
    def _remove_relevant_amber_caps(self,
                                    chemical_loss: ChemicalLoss,
                                    ) -> Chem.Mol:
        """
        Removes the relevant amber caps from the structure and returns a deep copy.

        TODO: implement relevant logic here.

        Args:
            chemical_loss: The chemical loss object containing atom indices.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        return self.initial_rdkit_mol

    def _make_structure(self, 
                        chemical_loss: ChemicalLoss,
                        ) -> Chem.Mol:
        """
        Make a structure given a ChemicalLoss. Equivalent to applying the 
        ChemicalLoss to the initial structure (as if its supposed 
        bond chemistry were actually present).
        """
        try:
            modifier_type = self._chem_loss_to_modifier_type[chemical_loss.method]
        except KeyError:
            raise ValueError(f"No modifier type found for chemical loss method: {chemical_loss.method}")
        
        if self._is_terminal_subclass(modifier_type):
            init_mol = self._remove_relevant_amber_caps(chemical_loss)
        else:
            init_mol = self.initial_rdkit_mol
        
        modifier = modifier_type(init_mol)
        return modifier.modify_structure(chemical_loss, 
                                         self._mdtraj_atom_idxs_dict,
                                         self._rdkit_atom_idxs_dict,
                                         )
    
    def make_structure(self,
                       chemical_loss: ChemicalLoss,
                       positions: Optional[ArrayLike] = None,
                       ) -> Chem.Mol:
        """
        Make a structure given a ChemicalLoss.
        """
        if positions is not None:
            self._set_positions(positions, 
                                self._chemical_loss_handler.units_factor,
                                )
        return self._make_structure(chemical_loss)