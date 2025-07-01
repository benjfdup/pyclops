from typing import Dict, Type

from .loss_structure_modifier import LossStructureModifier
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
from .structure_modifier.carboxylic_carbo_modifier import (
    AspGluModifier,
    AspAspModifier,
    GluGluModifier,
    AspCTermModifier,
    GluCTermModifier,
)
from .structure_modifier.disulfide_modifier import DisulfideModifier

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

    # carboxylic carbo losses
    AspGluModifier._method: AspGluModifier,
    AspAspModifier._method: AspAspModifier,
    GluGluModifier._method: GluGluModifier,
    AspCTermModifier._method: AspCTermModifier,
    GluCTermModifier._method: GluCTermModifier,

    # disulfide losses
    DisulfideModifier._method: DisulfideModifier,
}