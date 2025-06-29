from abc import ABC, abstractmethod
from typing import final, Dict, Tuple

from rdkit import Chem

from ...core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict

class LossStructureModifier(ABC):
    """
    Abstract base class for loss structure modifiers.

    Be sure to handle resonance properly.
    """
    _method: str = ''

    def __init__(self,):
        if self._method == '':
            raise ValueError('Subclasses must define a _method attribute')
    
    def _reverse_atom_indexes_dict(self,
                                   atom_indexes_dict: AtomIndexDict,
                                   ) -> Dict[int, Tuple[int, str]]:
        """
        Reverse the atom indexes dictionary.
        """
        # this only works because the mapping is 1-to-1
        return {v: k for k, v in atom_indexes_dict.items()}
    
    @abstractmethod
    def _mod_struct(self,
                    chemical_loss: ChemicalLoss,
                    rdkit_mol: Chem.Mol,
                    residue_idx_atom_name_to_atom_idx: AtomIndexDict,
                    atom_indexes_dict: AtomIndexDict, # from ChemicalLossHandler
                    ) -> Chem.Mol:
        """
        Modify the structure according to the provided ChemicalLoss.
        This is the method that subclasses must implement. It should be used
        internally by the `modify_structure` method.

        Args:
            chemical_loss: The ChemicalLoss to modify the structure according to.
            rdkit_mol: The RDKit molecule to modify.
            residue_idx_atom_name_to_atom_idx: A dictionary mapping from (residue_index: int, atom_name: str) 
            to atom indexes (int) in the RDKit molecule. Note that this is different from the atom_indexes_dict.
            atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indices (int)
            in the ChemicalLossHandler. Note that this is different from the residue_idx_atom_name_to_atom_idx.
        """
        raise NotImplementedError('Subclasses must implement this method')
    
    @final
    def _validate_chemical_loss(self,
                                chemical_loss: ChemicalLoss,
                                ) -> None:
        """
        Validate the chemical loss.
        """
        if chemical_loss._resonance_key[0] != self._method:
            raise ValueError(f'ChemicalLoss of method {chemical_loss._method} is not valid for a '
                             f'LossStructureModifier expecting a {self._method} chemical loss')

    @final
    def modify_structure(self,
                         chemical_loss: ChemicalLoss,
                         rdkit_mol: Chem.Mol,
                         residue_idx_atom_name_to_atom_idx: AtomIndexDict,
                         atom_indexes_dict: AtomIndexDict, # from ChemicalLossHandler
                         ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.

        Args:
            chemical_loss: The ChemicalLoss to modify the structure according to.
            rdkit_mol: The RDKit molecule to modify.
            residue_idx_atom_name_to_atom_idx: A dictionary mapping from (residue_index: int, atom_name: str) 
            to atom indexes (int) in the RDKit molecule. Note that this is different from the atom_indexes_dict.
            atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indices (int)
            in the ChemicalLossHandler. Note that this is different from the residue_idx_atom_name_to_atom_idx.
        Returns:
            The modified RDKit molecule.
        """
        self._validate_chemical_loss(chemical_loss)
        return self._mod_struct(chemical_loss, 
                                rdkit_mol, 
                                residue_idx_atom_name_to_atom_idx, 
                                atom_indexes_dict,
                                )