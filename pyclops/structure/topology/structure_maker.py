"""
This module contains the StructureMaker class, 
which is used to make a structure given an initial structure and a ChemicalLoss.
"""
__all__ = [
    "StructureMaker",
]

from typing import Union, Dict, Tuple

import numpy as np
import torch
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import BondType

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

        This method uses MDAnalysis's built-in RDKit conversion functionality
        or falls back to PDB-based conversion if direct conversion isn't available.

        Returns:
            Chem.Mol, the RDKit molecule
        """
        # Use MDAnalysis's built-in RDKit conversion
        ag = self._initial_universe.select_atoms("protein")
        return ag.to_rdkit()
    
    def _universe_to_rdkit_with_mapping(self,
                                       ) -> Tuple[Chem.Mol, Dict[Tuple[int, str], int]]:
        """
        Convert MDAnalysis universe to RDKit molecule with proper bond types
        and create a mapping from (residue_index, atom_name) to RDKit atom index.
        
        This method handles bond types correctly (e.g., C=O, S-S) and keeps hydrogens implicit.
        
        Returns:
            - RDKit Mol with correct bond types
            - Dictionary mapping (residue_index, atom_name) -> RDKit atom index
        """
        from rdkit import Chem
        from rdkit.Chem import BondType
        
        # Create editable molecule
        emol = Chem.EditableMol(Chem.Mol())
        
        # Maps for tracking indices
        mda_to_rdkit_idx: Dict[int, int] = {}
        resatom_to_rdkit_idx: Dict[Tuple[int, str], int] = {}
        
        # Get protein atoms only (exclude hydrogens)
        protein_atoms = self._initial_universe.select_atoms("protein and not name H*")
        
        # 1) Add all non-hydrogen atoms
        for atom in protein_atoms:
            res_idx = atom.resid
            atom_name = atom.name.strip()
            element = atom.element.strip().capitalize()
            
            if not element:
                raise ValueError(
                    f"Atom {atom} has no element assigned. "
                    "Consider running u.add_TopologyAttr('elements') or check your input."
                )
            
            rdkit_atom = Chem.Atom(element)
            emol_idx = emol.AddAtom(rdkit_atom)
            
            mda_to_rdkit_idx[atom.index] = emol_idx
            resatom_to_rdkit_idx[(res_idx, atom_name)] = emol_idx
        
        # 2) Add bonds with proper bond types
        for bond in self._initial_universe.bonds:
            # Get the two atoms in the bond
            atom1, atom2 = bond.atoms
            
            # Skip bonds involving hydrogens
            if atom1.name.startswith('H') or atom2.name.startswith('H'):
                continue
                
            # Get RDKit indices
            idx1 = mda_to_rdkit_idx.get(atom1.index)
            idx2 = mda_to_rdkit_idx.get(atom2.index)
            
            # Skip if either atom wasn't added (e.g., hydrogen)
            if idx1 is None or idx2 is None:
                continue
            
            # Determine bond type based on atom types and connectivity
            bond_type = self._determine_bond_type(atom1, atom2)
            
            emol.AddBond(idx1, idx2, bond_type)
        
        # 3) Convert to RDKit Mol
        mol = emol.GetMol()
        
        # 4) Sanitize (assign valences, update hydrogens, set aromaticity)
        Chem.SanitizeMol(mol)
        
        return mol, resatom_to_rdkit_idx
    
    def _determine_bond_type(self, atom1: mda.Atom, atom2: mda.Atom) -> BondType:
        """
        Determine the appropriate bond type between two atoms based on their
        chemical context and connectivity.
        
        Args:
            atom1, atom2: MDAnalysis atoms
            
        Returns:
            RDKit BondType
        """
        # Get atom names and elements
        name1, name2 = atom1.name.strip(), atom2.name.strip()
        elem1, elem2 = atom1.element.strip().upper(), atom2.element.strip().upper()
        
        # Carbonyl bonds (C=O)
        if (elem1 == 'C' and elem2 == 'O') or (elem1 == 'O' and elem2 == 'C'):
            # Check if this is a carbonyl carbon (connected to O)
            c_atom = atom1 if elem1 == 'C' else atom2
            o_atom = atom2 if elem2 == 'O' else atom1
            
            # Look for carbonyl patterns
            if self._is_carbonyl_carbon(c_atom):
                return BondType.DOUBLE
        
        # Disulfide bonds (S-S)
        if elem1 == 'S' and elem2 == 'S':
            # Check if these are cysteine SG atoms
            if name1 == 'SG' and name2 == 'SG':
                return BondType.SINGLE  # Disulfide is typically represented as single
        
        # Amide bonds (C-N in peptide backbone)
        if (elem1 == 'C' and elem2 == 'N') or (elem1 == 'N' and elem2 == 'C'):
            c_atom = atom1 if elem1 == 'C' else atom2
            n_atom = atom2 if elem2 == 'N' else atom1
            
            # Check if this is a peptide backbone amide
            if self._is_peptide_amide(c_atom, n_atom):
                return BondType.SINGLE  # Amide C-N is single
        
        # Default to single bond
        return BondType.SINGLE
    
    def _is_carbonyl_carbon(self, atom: mda.Atom) -> bool:
        """
        Check if a carbon atom is part of a carbonyl group (C=O).
        """
        # Look for connected oxygen atoms
        for bond in atom.bonds:
            other_atom = bond.atoms[0] if bond.atoms[1] == atom else bond.atoms[1]
            if other_atom.element.strip().upper() == 'O':
                return True
        return False
    
    def _is_peptide_amide(self, c_atom: mda.Atom, n_atom: mda.Atom) -> bool:
        """
        Check if a C-N bond is part of a peptide backbone amide.
        """
        # This is a simplified check - in practice you might want more sophisticated logic
        # Look for carbonyl oxygen connected to the carbon
        for bond in c_atom.bonds:
            other_atom = bond.atoms[0] if bond.atoms[1] == c_atom else bond.atoms[1]
            if other_atom.element.strip().upper() == 'O':
                return True
        return False
    
    def get_rdkit_atom_index(self, 
                            residue_index: int, 
                            atom_name: str,
                            mol: Chem.Mol = None,
                            mapping: Dict[Tuple[int, str], int] = None
                            ) -> int:
        """
        Get the RDKit atom index for a specific residue and atom name.
        
        Args:
            residue_index: Residue index in the MDAnalysis universe
            atom_name: Atom name (e.g., "NZ", "CA", "CB")
            mol: Optional RDKit molecule (if None, will convert universe)
            mapping: Optional mapping dict (if None, will be created)
            
        Returns:
            RDKit atom index
            
        Raises:
            KeyError: If the (residue_index, atom_name) combination is not found
        """
        if mol is None or mapping is None:
            mol, mapping = self._universe_to_rdkit_with_mapping()
        
        key = (residue_index, atom_name)
        if key not in mapping:
            raise KeyError(f"Atom {atom_name} in residue {residue_index} not found in mapping")
        
        return mapping[key]
    
    def get_rdkit_molecule_with_mapping(self) -> Tuple[Chem.Mol, Dict[Tuple[int, str], int]]:
        """
        Get the RDKit molecule and mapping dictionary.
        
        Returns:
            - RDKit Mol with correct bond types
            - Dictionary mapping (residue_index, atom_name) -> RDKit atom index
        """
        return self._universe_to_rdkit_with_mapping()

    def _make_structure(self, 
                        chemical_loss: ChemicalLoss,
                        ) -> mda.Universe:
        """
        Make a structure given a ChemicalLoss.
        """
        pass