from rdkit import Chem

from ....core.chemical_loss.chemical_loss import AtomIndexDict, ResonanceKey
from ....losses.disulfide import Disulfide
from ..loss_structure_modifier import LossStructureModifier

class DisulfideModifier(LossStructureModifier):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """

    _method = Disulfide._method
    pass