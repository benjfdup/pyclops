from abc import ABCMeta
from typing import final

from rdkit import Chem

from ..loss_structure_modifier import LossStructureModifier
from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.carboxylic_carbo import (
    AspGlu,
    AspAsp,
    GluGlu,
    AspCTerm,
    GluCTerm,
)

class CarboxylicCarboxylicModifier(LossStructureModifier, metaclass=ABCMeta):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _fragment: str = "c1cc(ccc1CN)CN" # the fragment to be added to the structure
    _fragment_rdkit_mol: Chem.Mol = Chem.MolFromSmiles(_fragment)

    @property
    def fragment_rdkit_mol(self) -> Chem.Mol:
        return Chem.Mol(self._fragment_rdkit_mol)
    
    @final
    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   o1_idx: int, # rdkit atom index
                   o2_idx: int, # rdkit atom index
                   ) -> Chem.Mol:
        """
        To be used in the `_outer_mod` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It also exists soley to organize
        sets of instructions (to make the codebase more readable and modular). It is called `_inner_mod`
        because it is fundamentally what actually modifies the structure. In a way, it is the workhorse
        of all `CarboxylicCarboxylicModifier` subclasses.

        Args:
            initial_parsed_mol: The initial parsed molecule.
            o1_idx: The index of the first oxygen (rdkit atom index) - to be converted to nitrogen.
            o2_idx: The index of the second oxygen (rdkit atom index) - to be converted to nitrogen.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        emol = Chem.EditableMol(initial_parsed_mol)

        if o1_idx > o2_idx:
            greater_idx = o1_idx
            lesser_idx = o2_idx
        elif o1_idx < o2_idx:
            greater_idx = o2_idx
            lesser_idx = o1_idx
        else:
            raise ValueError(f"Oxygen indices {o1_idx} and {o2_idx} are the same")
        
        # Get the fragment molecule and its atoms
        fragment_mol = self.fragment_rdkit_mol
        fragment_atoms = fragment_mol.GetAtoms()
        
        # Find the two nitrogen atoms in the fragment
        nitrogen_atoms = [atom for atom in fragment_atoms if atom.GetAtomicNum() == 7]
        if len(nitrogen_atoms) != 2:
            raise ValueError(f"Expected 2 nitrogen atoms in fragment, found {len(nitrogen_atoms)}")
        
        n1_idx = nitrogen_atoms[0].GetIdx()
        n2_idx = nitrogen_atoms[1].GetIdx()
        
        # Get the original molecule's atoms
        mol_atoms = initial_parsed_mol.GetAtoms()
        o1_atom = mol_atoms[o1_idx]
        o2_atom = mol_atoms[o2_idx]
        
        # Check that oxygens are only bound to carbon atoms
        for o_idx, o_atom in [(o1_idx, o1_atom), (o2_idx, o2_atom)]:
            for bond in o_atom.GetBonds():
                other_atom_idx = bond.GetOtherAtomIdx(o_idx)
                other_atom = mol_atoms[other_atom_idx]
                if other_atom.GetAtomicNum() != 6:  # 6 is carbon
                    raise ValueError(f"Oxygen at index {o_idx} is bound to non-carbon atom {other_atom.GetSymbol()} at index {other_atom_idx}")
        
        # Store the bonds that need to be recreated
        o1_bonds = [(bond.GetOtherAtomIdx(o1_idx), bond.GetBondType()) 
                    for bond in o1_atom.GetBonds()]
        o2_bonds = [(bond.GetOtherAtomIdx(o2_idx), bond.GetBondType()) 
                    for bond in o2_atom.GetBonds()]
        
        # Remove the oxygen atoms
        emol.RemoveAtom(o1_idx)
        emol.RemoveAtom(o2_idx)
        
        # Add the fragment to the molecule
        combined_mol = Chem.CombineMols(initial_parsed_mol, fragment_mol)
        emol = Chem.EditableMol(combined_mol)
        
        # Get the new indices after adding the fragment
        # The fragment atoms will be at the end of the molecule
        fragment_start_idx = len(mol_atoms)
        new_n1_idx = fragment_start_idx + n1_idx
        new_n2_idx = fragment_start_idx + n2_idx
        
        # Recreate the bonds for the first nitrogen (replacing greater_idx oxygen)
        for bond_idx, bond_type in o1_bonds:
            # Adjust bond_idx if it was affected by the removal
            adjusted_bond_idx = bond_idx
            if bond_idx > greater_idx:
                adjusted_bond_idx -= 1
            if bond_idx > lesser_idx:
                adjusted_bond_idx -= 1
            
            emol.AddBond(adjusted_bond_idx, new_n1_idx, bond_type)
        
        # Recreate the bonds for the second nitrogen (replacing lesser_idx oxygen)
        for bond_idx, bond_type in o2_bonds:
            # Adjust bond_idx if it was affected by the removal
            adjusted_bond_idx = bond_idx
            if bond_idx > greater_idx:
                adjusted_bond_idx -= 1
            if bond_idx > lesser_idx:
                adjusted_bond_idx -= 1
            
            emol.AddBond(adjusted_bond_idx, new_n2_idx, bond_type)
        
        return emol.GetMol()
    
    @final
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ) -> Chem.Mol:
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which oxygen to remove when forming a carboxylic acid, then calls `_inner_mod` to perform the actual
        structural modification.
        
        Args:
            chemical_loss: The chemical loss object containing atom indices.
            initial_parsed_mol: The initial parsed molecule.
            mdtraj_atom_indexes_dict: Dictionary mapping atom names to MDTraj atom indices.
            rdkit_atom_indexes_dict: Dictionary mapping atom names to RDKit atom indices.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        o1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O1']]
        o2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O2']]

        o1_rdkit_idx = rdkit_atom_indexes_dict[o1_key]
        o2_rdkit_idx = rdkit_atom_indexes_dict[o2_key]
        
        return self._inner_mod(initial_parsed_mol, o1_rdkit_idx, o2_rdkit_idx)
    
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        """
        return self._outer_mod(
            chemical_loss, 
            initial_parsed_mol, 
            mdtraj_atom_indexes_dict, 
            rdkit_atom_indexes_dict,
            )
    
class CarboxylicCarboSide2SideModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a sidechain carboxyl group.
    """
    pass

class AspGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with Glutamate's carboxyl group.
    """
    _method = AspGlu._method

class AspAspModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with itself.
    """
    _method = AspAsp._method

class GluGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Glutamate's carboxyl group forms a bond with itself.
    """
    _method = GluGlu._method

class CarboxylicCarbo2CTermModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    pass

class AspCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Aspartate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = AspCTerm._method

class GluCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Glutamate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = GluCTerm._method