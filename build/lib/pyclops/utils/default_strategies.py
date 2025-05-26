from typing import Set, Type

from ..core.chemical_loss import ChemicalLoss
from ..losses.amide_losses import *
from ..losses.carboxylic_carbo import *
from ..losses.cysteine_carbo import *
from ..losses.disulfide import *
from ..losses.lys_arg import *
from ..losses.lys_tyr import *


DEFAULT_STRATEGIES: Set[Type[ChemicalLoss]] = {
    # amide:
    AmideHead2Tail,
    AmideLysGlu,
    AmideLysAsp,
    AmideOrnGlu,
    AmideLysHead,
    AmideArgHead,
    AmideLysTail,
    AmideOrnTail,

    # carboxylic-carbo
    AspGlu,
    AspCTerm,
    GluCTerm,
    AspAsp,
    GluGlu,
    
    # cysteine-carbo
    CysCTerm,
    CysAsp,
    CysGlu,

    # disulfide
    Disulfide,

    # lys-arg
    LysArg,

    # lys-tyr
    LysTyr,
}