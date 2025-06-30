from typing import Dict, Type

from .loss_structure_modifier import LossStructureModifier
from .structure_modifier.disulfide_modifier import DisulfideModifier

DEFAULT_MODIFIER_DICT: Dict[str, Type[LossStructureModifier]] = {
    DisulfideModifier._method: DisulfideModifier,
}