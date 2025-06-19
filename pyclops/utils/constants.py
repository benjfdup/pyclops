from typing import Set, Dict

# Boltzmann constant in kcal/mol/K - more appropriate for molecular energy calculations
KB: float = 1.0  # kcal/mol/K

AMBER_CAPS_SMILES_DICT: Dict[str, str] = { # this is essentially just for organizational purposes
    "ACE": "CC(=O)", # N terminal cap
    "NME": "NC", # C terminal cap
    "NHE": "N", # C terminal cap
}
AMBER_CAPS: Set[str] = set(AMBER_CAPS_SMILES_DICT.keys()) # cap amino acids used in amber md simulations
AMBER_CAPS_SMILES: Set[str] = set(AMBER_CAPS_SMILES_DICT.values())
UNITS_FACTORS_DICT: Dict[str, float] = {
    "angstrom": 1.0,
     "A": 1.0, # alias

    "nanometer": 10.0,
    "nm": 10.0, # alias

    "picometer": 0.01,
    "pm": 0.01, # alias
    }