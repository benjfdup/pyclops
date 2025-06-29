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
                    chemical_loss: ChemicalLoss,
                    rdkit_mol: Chem.Mol,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.
        """
        mdtraj_idx_to_rdkit_idx = self._mdtraj_idx_to_rdkit_idx_dict(
            mdtraj_atom_indexes_dict,
            rdkit_atom_indexes_dict,
        )
        loss_atom_idxs = chemical_loss._atom_idxs
        s1_idx_mdtraj = loss_atom_idxs['S1']
        s2_idx_mdtraj = loss_atom_idxs['S2']

        s1_idx_rdkit = mdtraj_idx_to_rdkit_idx[s1_idx_mdtraj]
        s2_idx_rdkit = mdtraj_idx_to_rdkit_idx[s2_idx_mdtraj]
        
        # Add the disulfide bond between the two sulfur atoms
        rdkit_mol.AddBond(s1_idx_rdkit, s2_idx_rdkit, Chem.BondType.SINGLE)
        Chem.SanitizeMol(rdkit_mol)  # Optional but recommended

        return rdkit_mol