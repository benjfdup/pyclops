from abc import ABCMeta
from typing import final

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign

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
    
    @final
    def _inner_mod(self,
                initial_parsed_mol: Chem.Mol,
                o1_idx: int,
                o2_idx: int,
                ) -> Chem.Mol:
        """
        Inner functionality to modify the structure according to the corresponding ChemicalLoss.
        """
        
        # Store the original number of atoms to identify which are new
        original_num_atoms = initial_parsed_mol.GetNumAtoms() # only works because we never remove atoms
        
        emol = Chem.EditableMol(initial_parsed_mol)

        # Create nitrogen atoms to replace the oxygens
        n1 = Chem.Atom('N')
        n2 = Chem.Atom('N')
        
        # Replace the oxygen atoms with nitrogen atoms
        emol.ReplaceAtom(o1_idx, n1)
        emol.ReplaceAtom(o2_idx, n2)

        outer_1 = emol.AddAtom(Chem.Atom('C'))
        outer_2 = emol.AddAtom(Chem.Atom('C'))

        emol.AddBond(outer_1, o1_idx, Chem.BondType.SINGLE)
        # NOTE: NOT adding the bond between outer_2 and o2_idx here - saving for last

        bc1 = emol.AddAtom(Chem.Atom('C')) # bonded to outer_1
        bc2 = emol.AddAtom(Chem.Atom('C'))
        bc3 = emol.AddAtom(Chem.Atom('C'))
        bc4 = emol.AddAtom(Chem.Atom('C')) # bounded to outer_2
        bc5 = emol.AddAtom(Chem.Atom('C'))
        bc6 = emol.AddAtom(Chem.Atom('C'))

        emol.AddBond(bc1, bc2, Chem.BondType.AROMATIC)
        emol.AddBond(bc2, bc3, Chem.BondType.AROMATIC)
        emol.AddBond(bc3, bc4, Chem.BondType.AROMATIC)
        emol.AddBond(bc4, bc5, Chem.BondType.AROMATIC)
        emol.AddBond(bc5, bc6, Chem.BondType.AROMATIC)
        emol.AddBond(bc6, bc1, Chem.BondType.AROMATIC)

        emol.AddBond(bc1, outer_1, Chem.BondType.SINGLE)
        emol.AddBond(bc4, outer_2, Chem.BondType.SINGLE)  # Add this bond now
        
        # Get the intermediate molecule and sanitize it
        intermediate_mol = emol.GetMol()
        Chem.SanitizeMol(intermediate_mol)

        # Now add the final bond between outer_2 and o2_idx (which is now nitrogen)
        emol_final = Chem.EditableMol(intermediate_mol)
        emol_final.AddBond(outer_2, o2_idx, Chem.BondType.SINGLE)
        
        # Get the final molecule and sanitize it
        final_mol = emol_final.GetMol()
        
        # Create coordinate map to constrain existing atoms to their original positions
        coord_map = {}
        original_conf = initial_parsed_mol.GetConformer()
        
        # Constrain all original atoms (including the replaced N atoms at O positions)
        for i in range(original_num_atoms):
            coord_map[i] = original_conf.GetAtomPosition(i)
        
        # Use ETKDG with coordinate constraints
        rdDistGeom.EmbedMolecule(final_mol, 
                            coordMap=coord_map,
                            randomSeed=42,
                            useRandomCoords=False)
        Chem.SanitizeMol(final_mol)
        
        return final_mol
    
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