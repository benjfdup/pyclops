from abc import ABCMeta
from typing import final, Optional

from rdkit import Chem

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..loss_structure_modifier import LossStructureModifier
from ....losses.amide_losses import (AmideHead2Tail, 
                                     AmideLysGlu, 
                                     AmideLysAsp, 
                                     AmideOrnGlu, 
                                     AmideAspHead, 
                                     AmideGluHead, 
                                     AmideLysTail, 
                                     AmideArgTail, 
                                     AmideOrnTail,
                                     )


class AmideModifier(LossStructureModifier, metaclass=ABCMeta):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    Assumes no Amber Caps are present (as this makes handling the amide bond
    more complicated) -- Amber Caps will be removed in the StructureMaker.
    """
    @final
    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   carboxyl_carbon_idx: int,
                   amide_nitrogen_idx: int,
                   oxygen_to_remove_idx: Optional[int] = None,
                   ) -> Chem.Mol:
        """
        To be used in the `_outer_mod` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It also exists soley to organize
        sets of instructions (to make the codebase more readable and modular). It is called `_inner_mod`
        because it is fundamentally what actually modifies the structure. In a way, it is the workhorse
        of all `AmideModifier` subclasses.
        
        Args:
            initial_parsed_mol: The initial parsed molecule.
            carboxyl_carbon_idx: The index of the carboxyl carbon (rdkit atom index).
            amide_nitrogen_idx: The index of the amide nitrogen (rdkit atom index).
            oxygen_to_remove_idx: The index of the oxygen to remove (rdkit atom index). 
            If None, no oxygen is removed.
            
        Returns:
            Modified RDKit molecule with amide bond formed
        """
        emol = Chem.EditableMol(initial_parsed_mol)

        emol.AddBond(carboxyl_carbon_idx, amide_nitrogen_idx, order=Chem.rdchem.BondType.SINGLE)
        if oxygen_to_remove_idx is not None:
            emol.RemoveAtom(oxygen_to_remove_idx)

        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)

        return new_mol
    
    @final
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   carboxyl_carbon_idx: int,
                   amide_nitrogen_idx: int,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ):
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which oxygen to remove when forming an amide bond, then calls `_inner_mod` to perform the actual
        structural modification.
        
        Args:
            chemical_loss: The chemical loss object containing atom indices.
            initial_parsed_mol: The initial parsed molecule.
            carboxyl_carbon_idx: The index of the carboxyl carbon (mdtraj atom index).
            amide_nitrogen_idx: The index of the amide nitrogen (mdtraj atom index).
            mdtraj_atom_indexes_dict: Dictionary mapping atom names to MDTraj atom indices.
            rdkit_atom_indexes_dict: Dictionary mapping atom names to RDKit atom indices.
            
        Returns:
            Modified RDKit molecule with amide bond formed
        """

        # get all oxygens attached to the carboxyl carbon
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        carboxyl_carbon_key = inverse_mdtraj_atom_indexes_dict[carboxyl_carbon_idx]
        amide_nitrogen_key = inverse_mdtraj_atom_indexes_dict[amide_nitrogen_idx]

        carboxyl_carbon_rdkit_idx = rdkit_atom_indexes_dict[carboxyl_carbon_key]
        amide_nitrogen_rdkit_idx = rdkit_atom_indexes_dict[amide_nitrogen_key]

        carboxyl_carbon = initial_parsed_mol.GetAtomWithIdx(carboxyl_carbon_rdkit_idx)
        oxygen_indices = [] # rdkit atom indices
        
        for neighbor in carboxyl_carbon.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:  # Atomic number 8 is oxygen
                oxygen_indices.append(neighbor.GetIdx())

        if len(oxygen_indices) > 2:
            raise ValueError(f"Carboxyl carbon {carboxyl_carbon_idx} has {len(oxygen_indices)} oxygens attached to it, "
                             f"but only 2 are allowed for an amide bond.")
        if len(oxygen_indices) == 0:
            raise ValueError(f"Carboxyl carbon {carboxyl_carbon_idx} has no oxygens attached to it")
        
        if len(oxygen_indices) == 1:
            return self._inner_mod(initial_parsed_mol,
                                   carboxyl_carbon_rdkit_idx,
                                   amide_nitrogen_rdkit_idx,
                                   oxygen_to_remove_idx=None,
                                   )
        else:
            oxygen_to_keep_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O1']]
            oxygen_to_keep_rdkit_idx = rdkit_atom_indexes_dict[oxygen_to_keep_key]

            if oxygen_to_keep_rdkit_idx not in oxygen_indices:
                raise ValueError(f"Oxygen to keep {oxygen_to_keep_rdkit_idx} is not attached to the carboxyl carbon {carboxyl_carbon_idx}")

            # remove the oxygen that is not the one to keep
            oxygen_to_remove_idx = oxygen_indices[0] if oxygen_indices[0] != oxygen_to_keep_rdkit_idx else oxygen_indices[1]

            return self._inner_mod(
                initial_parsed_mol,
                carboxyl_carbon_rdkit_idx,
                amide_nitrogen_rdkit_idx,
                oxygen_to_remove_idx=oxygen_to_remove_idx,
                )
    
    # subclasses may override this method, but the default 
    # implementation is provided here for convenience.
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.
        """
        return self._outer_mod(chemical_loss,
                               initial_parsed_mol,
                               carboxyl_carbon_idx=chemical_loss._atom_idxs['C1'],
                               amide_nitrogen_idx=chemical_loss._atom_idxs['N1'],
                               mdtraj_atom_indexes_dict=mdtraj_atom_indexes_dict,
                               rdkit_atom_indexes_dict=rdkit_atom_indexes_dict,)

class Amide2TermModifier(AmideModifier):
    """
    An abstract base class for amide modifiers that involve either the N-terminus, 
    the C-terminus, or both.

    Note: subclasses are not compatible with Amber Caps (though which one needs to be removed
    depends on which head is being bonded to). Regardless, prerequisite modification (ie the removal
    of the relevant Amber Cap) must be performed in the `StructureMaker` class (or at least outside of
    the `AmideModifier` subclasses).
    """
    pass

@final
class AmideHead2TailModifier(Amide2TermModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideHead2Tail._method
    
    
class AmideSide2SideModifier(AmideModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    Exists solely as an organizational placeholder.
    """
    pass


@final
class AmideLysGluModifier(AmideSide2SideModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideLysGlu._method
    
    
@final
class AmideLysAspModifier(AmideSide2SideModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideLysAsp._method
    
    
@final
class AmideOrnGluModifier(AmideSide2SideModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideOrnGlu._method


class AmideSide2HeadModifier(Amide2TermModifier):
    """
    Base class for amide modifiers that connect a sidechain to the N-terminus.
    """
    pass


@final
class AmideAspHeadModifier(AmideSide2HeadModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideAspHead._method
    

@final
class AmideGluHeadModifier(AmideSide2HeadModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideGluHead._method
    

class AmideSide2TailModifier(Amide2TermModifier):
    """
    Base class for amide modifiers that connect a sidechain to the C-terminus.
    """
    pass


@final
class AmideLysTailModifier(AmideSide2TailModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideLysTail._method
    
    
@final
class AmideArgTailModifier(AmideSide2TailModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideArgTail._method
    

@final
class AmideOrnTailModifier(AmideSide2TailModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _method = AmideOrnTail._method