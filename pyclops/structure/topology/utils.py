from typing import Dict, Type

from .loss_structure_modifier import LossStructureModifier
from .structure_modifier.disulfide_modifier import DisulfideModifier
from .structure_modifier.amide_modifier import (
    AmideHead2TailModifier,
    AmideLysGluModifier,
    AmideLysAspModifier,
    AmideOrnGluModifier,
    AmideAspHeadModifier,
    AmideGluHeadModifier,
    AmideLysTailModifier,
    AmideArgTailModifier,
    AmideOrnTailModifier,
)

DEFAULT_MODIFIER_DICT: Dict[str, Type[LossStructureModifier]] = {
    # amide losses
    AmideHead2TailModifier._method: AmideHead2TailModifier,
    AmideLysGluModifier._method: AmideLysGluModifier,
    AmideLysAspModifier._method: AmideLysAspModifier,
    AmideOrnGluModifier._method: AmideOrnGluModifier,
    AmideAspHeadModifier._method: AmideAspHeadModifier,
    AmideGluHeadModifier._method: AmideGluHeadModifier,
    AmideLysTailModifier._method: AmideLysTailModifier,
    AmideArgTailModifier._method: AmideArgTailModifier,
    AmideOrnTailModifier._method: AmideOrnTailModifier,

    # disulfide losses
    DisulfideModifier._method: DisulfideModifier,
}