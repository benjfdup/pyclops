from abc import ABC, abstractmethod
from typing import final, Dict

from rdkit import Chem

from ...core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict

class LossStructureModifier(ABC):
    """
    Abstract base class for loss structure modifiers.

    Be sure to handle resonance properly.
    """
    _method: str = ''

    def __init__(self,
                 initial_parsed_mol: Chem.Mol, # should be produced by MDAnalysis's to_rdkit() method
                 ):
        """
        Args:
            initial_parsed_mol: RDKit molecule, produced by MDAnalysis's to_rdkit() method.
                This should come with info on residues and atom idxs (which is not the default in RDKit).
        """
        if self._method == '':
            raise ValueError('Subclasses must define a _method attribute')
        self._initial_parsed_mol = initial_parsed_mol # must come  with info on residues and atom idxs.

    @final
    @property
    def initial_parsed_mol(self) -> Chem.Mol:
        """
        A deep copy of the initial parsed molecule.
        """
        return Chem.Mol(self._initial_parsed_mol) # clones the initial parsed molecule
                                                  # Note: we cannot use deepcopy here, as RDKit molecules are C++ objects
                                                  # under the hood. Special thanks to Dr. Richard Lewis: 
                                                  # https://sourceforge.net/p/rdkit/mailman/message/33652439/
    
    @final
    @staticmethod
    def _invert_dict(
        original_dict: Dict,
    ) -> Dict:
        """
        Invert a dictionary, ensuring no repeated values exist.
        
        Raises:
            ValueError: If the original dictionary contains duplicate values
                    that would cause key collisions in the inverted dictionary.
        """
        values = list(original_dict.values())
        if len(values) != len(set(values)):
            raise ValueError("Cannot invert dictionary: contains duplicate values that would cause key collisions")
        
        return {v: k for k, v in original_dict.items()}

    @abstractmethod
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the provided ChemicalLoss.
        This is the method that subclasses must implement. It should be used
        internally by the `modify_structure` method.

        Args:
            initial_parsed_mol: The initial parsed molecule.
            chemical_loss: The ChemicalLoss to modify the structure according to.
            mdtraj_atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indices (int) in the ChemicalLossHandler.
            rdkit_atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indexes (int) in the RDKit molecule.

        Returns:
            The modified RDKit molecule (based on the initial parsed molecule)
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
                         mdtraj_atom_indexes_dict: AtomIndexDict,
                         rdkit_atom_indexes_dict: AtomIndexDict,
                         ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.

        Args:
            chemical_loss: The ChemicalLoss to modify the structure according to.
            mdtraj_atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indices (int) in the ChemicalLossHandler. Note that this is different from the residue_idx_atom_name_to_atom_idx.
            rdkit_atom_indexes_dict: A dictionary mapping from (residue_index: int, atom_name: str) to atom indexes (int) in the RDKit molecule. Note that this is different from the atom_indexes_dict.
        Returns:
            The modified RDKit molecule.
        """
        self._validate_chemical_loss(chemical_loss)
        final_rdkit_mol = self._mod_struct(self.initial_parsed_mol,
                                           chemical_loss, 
                                           mdtraj_atom_indexes_dict,
                                           rdkit_atom_indexes_dict,)
        Chem.SanitizeMol(final_rdkit_mol)
        return final_rdkit_mol