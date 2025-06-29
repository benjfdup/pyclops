from rdkit import Chem

from ....core.chemical_loss.chemical_loss import AtomIndexDict, ResonanceKey
from ....losses.disulfide import Disulfide
from ..loss_structure_modifier import LossStructureModifier

class DisulfideModifier(LossStructureModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """

    _method = Disulfide._method

    def _mod_struct(self,
                    resonance_key: ResonanceKey,
                    rdkit_mol: Chem.Mol,
                    residue_idx_atom_name_to_atom_idx: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.
        """
        pass