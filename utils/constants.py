from typing import Set, Dict


KB: float = 1.380649 * 10**-23
AMBER_CAPS: Set[str] = {"ACE", "NME", "NHE"} # cap amino acids used in amber md simulations
UNITS_FACTORS_DICT: Dict[str, float] = {
    "angstrom": 1.0,
     "A": 1.0, # alias

    "nanometer": 10.0,
    "nm": 10.0, # alias

    "picometer": 0.01,
    "pm": 0.01, # alias
    }