"""
Default strategies for the ChemicalLossHandler. 
Includes all the losses that are to be considered standard by PyCLOPS.
"""

__all__ = ['DEFAULT_STRATEGIES']
from typing import Set, Type

from ...core.chemical_loss import ChemicalLoss
from ..amide_losses import *
from ..carboxylic_carbo import *
from ..cysteine_carbo import *
from ..disulfide import *
from ..lys_arg import *
from ..lys_tyr import *
from ..methionine_carbo import *
from ..nit_phe import *


DEFAULT_STRATEGIES: Set[Type[ChemicalLoss]] = {
    # amide:
    AmideHead2Tail,
    AmideLysGlu,
    AmideLysAsp,
    AmideAspHead,
    AmideGluHead,
    AmideLysTail,
    AmideArgTail,

    # carboxylic-carbo
    AspGlu,
    AspAsp,
    GluGlu,
    AspCTerm,
    GluCTerm,
    
    # cysteine-carbo
    CysAsp,
    CysGlu,
    CysCTerm,

    # methionine-carbo
    MetAsp,
    MetGlu,
    MetCTerm,

    # disulfide
    Disulfide,

    # lys-arg
    LysArg,

    # lys-tyr
    LysTyr,

    # nit-phe
    PheHead,
    LysPhe,
}

# default lookup table for method_str -> strategy
DEFAULT_METHOD_TO_STRATEGY: Dict[str, Type[ChemicalLoss]] = {strategy._method: strategy for strategy in DEFAULT_STRATEGIES}