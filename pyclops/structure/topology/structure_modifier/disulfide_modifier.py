from typing import final

from rdkit import Chem

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.disulfide import Disulfide
from ..loss_structure_modifier import LossStructureModifier


@final
class DisulfideModifier(LossStructureModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    Should only be used implicitly, inside of structure maker, but left public
    for testing and modularity.
    """
    _method = Disulfide._method
    
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.
        """
        emol = Chem.EditableMol(initial_parsed_mol)
        
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        s1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['S1']]
        s2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['S2']]

        s1_idx_rdkit = rdkit_atom_indexes_dict[s1_key]
        s2_idx_rdkit = rdkit_atom_indexes_dict[s2_key]
        
        # Add the disulfide bond between the two sulfur atoms
        emol.AddBond(s1_idx_rdkit, s2_idx_rdkit, order=Chem.rdchem.BondType.SINGLE)
        new_mol = emol.GetMol()

        Chem.SanitizeMol(new_mol)  # Optional but recommended

        return new_mol