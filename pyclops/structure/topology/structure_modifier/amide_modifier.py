from abc import ABCMeta
from typing import final

from rdkit import Chem

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.amide_losses import AmideHead2Tail
from ..loss_structure_modifier import LossStructureModifier


class AmideModifier(LossStructureModifier, metaclass=ABCMeta):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    Assumes no Amber Caps are present (as this makes handling the amide bond
    more complicated) -- Amber Caps will be removed in the StructureMaker.
    """

    def _inner_mod(self,
                   oxygen_to_remove_idx: int,
                   carboxyl_carbon_idx: int,
                   amide_nitrogen_idx: int,
                   ) -> Chem.Mol:
        """
        To be used in the _mod_struct method of the subclasses for utility purposes.
        
        Args:
            oxygen_to_remove_idx: Index of the oxygen atom to remove from the carboxyl group
            carboxyl_carbon_idx: Index of the carboxyl carbon atom
            amide_nitrogen_idx: Index of the amide nitrogen atom
            
        Returns:
            Modified RDKit molecule with amide bond formed
        """
        initial_rdkit_mol = self._initial_parsed_mol.Copy()
        
        # Remove the oxygen atom from the carboxyl group
        initial_rdkit_mol.RemoveAtom(oxygen_to_remove_idx)
        
        # Form the amide bond between carboxyl carbon and amide nitrogen
        initial_rdkit_mol.AddBond(carboxyl_carbon_idx, amide_nitrogen_idx, Chem.BondType.SINGLE)
        
        # Sanitize the molecule to ensure proper valence and aromaticity
        Chem.SanitizeMol(initial_rdkit_mol)
        
        return initial_rdkit_mol

        