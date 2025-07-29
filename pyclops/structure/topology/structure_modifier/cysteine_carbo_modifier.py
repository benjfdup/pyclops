from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.cysteine_carbo import (
    CysAsp,
    CysGlu,
    CysCTerm,
)
from ..loss_structure_modifier import LossStructureModifier

class CysteineCarboModifier(LossStructureModifier):
    """
    Base class for cysteine-carboxyl chemical interactions in protein structures.
    """
    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   s1_idx: int, # rdkit atom index, sulfur of the cysteine
                   c3_idx: int, # rdkit atom index, carboxyl carbon
                   oxygen_to_remove_idx: Optional[int] = None, # rdkit atom index, oxygen to remove
                   ) -> Chem.Mol:
        """
        To be used in the `_outer_mod` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It also exists soley to organize
        sets of instructions (to make the codebase more readable and modular). It is called `_inner_mod`
        because it is fundamentally what actually modifies the structure. In a way, it is the workhorse
        of all `CysteineCarboModifier` subclasses.

        Args:
            initial_parsed_mol: The initial parsed molecule.
            s1_idx: The index of the sulfur of the cysteine (rdkit atom index).
            c3_idx: The index of the carboxyl carbon (rdkit atom index).
            oxygen_to_remove_idx: The index of the oxygen to remove (rdkit atom index).

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        emol = Chem.EditableMol(initial_parsed_mol)

        # Step 1: Add a new carbon atom and bond it to the sulfur
        new_carbon_idx = emol.AddAtom(Chem.Atom('C'))
        emol.AddBond(s1_idx, new_carbon_idx, Chem.BondType.SINGLE)

        # Step 2: Bond the new carbon to the carboxyl carbon
        # Adding atoms doesn't affect existing indices, so c3_idx remains valid
        emol.AddBond(new_carbon_idx, c3_idx, Chem.BondType.SINGLE)

        # Step 3: Remove the oxygen if specified
        # Removing atoms shifts all subsequent indices down by 1
        if oxygen_to_remove_idx is not None:
            emol.RemoveAtom(oxygen_to_remove_idx)

        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)

        # Step 4: Relax the newly added carbon position
        self._relax_new_carbon_position(new_mol, new_carbon_idx)

        return new_mol
    
    def _relax_new_carbon_position(self, mol: Chem.Mol, new_carbon_idx: int):
        """
        Relax the position of the newly added carbon atom using ETKDG with coordinate constraints.
        Freezes all atoms except the newly added carbon and only optimizes that specific carbon's position.
        
        Args:
            mol: The RDKit molecule
            new_carbon_idx: Index of the newly added carbon
        """
        
        # Get 3D coordinates
        conf = mol.GetConformer()
        if conf.Is3D():
            # Create coordinate map to constrain all atoms except the new carbon
            coord_map = {}
            for i in range(mol.GetNumAtoms()):
                if i != new_carbon_idx:
                    coord_map[i] = conf.GetAtomPosition(i)
            
            # Use ETKDG with coordinate constraints to freeze all atoms except the new carbon
            rdDistGeom.EmbedMolecule(mol, 
                                    coordMap=coord_map,
                                    randomSeed=42,
                                    useRandomCoords=False)
    
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
        s1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['S1']]
        c3_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['C3']]

        s1_rdkit_idx = rdkit_atom_indexes_dict[s1_key]
        c3_rdkit_idx = rdkit_atom_indexes_dict[c3_key]

        # get all oxygens attached to the carboxyl carbon
        carboxyl_carbon = initial_parsed_mol.GetAtomWithIdx(c3_rdkit_idx)
        oxygen_indices = [] # rdkit atom indices

        for neighbor in carboxyl_carbon.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:  # Atomic number 8 is oxygen
                oxygen_indices.append(neighbor.GetIdx())

        if len(oxygen_indices) > 2:
            raise ValueError(f"Carboxyl carbon {c3_rdkit_idx} has {len(oxygen_indices)} oxygens attached to it, "
                             f"but only 2 are allowed in a carboxylic acid.")
        if len(oxygen_indices) == 0:
            raise ValueError(f"Carboxyl carbon {c3_key} has no oxygens attached to it.")
        
        if len(oxygen_indices) == 1:
            return self._inner_mod(
                initial_parsed_mol, 
                s1_rdkit_idx, 
                c3_rdkit_idx, 
                oxygen_to_remove_idx = None,
                )
        else:
            oxygen_to_keep_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O1']]
            oxygen_to_keep_rdkit_idx = rdkit_atom_indexes_dict[oxygen_to_keep_key]

            if oxygen_to_keep_rdkit_idx not in oxygen_indices:
                raise ValueError(f"Oxygen to keep {oxygen_to_keep_rdkit_idx} is not attached to the carboxyl carbon {c3_key}")

            # remove the oxygen that is not the one to keep
            oxygen_to_remove_idx = oxygen_indices[0] if oxygen_indices[0] != oxygen_to_keep_rdkit_idx else oxygen_indices[1]

            return self._inner_mod(initial_parsed_mol, s1_rdkit_idx, c3_rdkit_idx, oxygen_to_remove_idx)
        
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        """
        return self._outer_mod(chemical_loss, 
                               initial_parsed_mol, 
                               mdtraj_atom_indexes_dict, 
                               rdkit_atom_indexes_dict)
    
class CysteineCarboSide2SideModifier(CysteineCarboModifier):
    """
    Base class for cysteine-carboxyl chemical interactions between 2 sidechains.
    """
    pass

class CysAspModifier(CysteineCarboSide2SideModifier):
    """
    Cysteine's carboxyl group forms a bond with an aspartic acid carboxyl group.
    """
    _method = CysAsp._method

class CysGluModifier(CysteineCarboSide2SideModifier):
    """
    Cysteine's carboxyl group forms a bond with a glutamic acid carboxyl group.
    """
    _method = CysGlu._method

class CysteineCarbo2CTermModifier(CysteineCarboModifier): # a bit overkill, but very good for the pattern.
    """
    Base class for cysteine-carboxyl chemical interactions between a sidechain and a c-terminal 
    carboxyl group.
    """
    pass

class CysCTermModifier(CysteineCarbo2CTermModifier):
    """
    Cysteine's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = CysCTerm._method
