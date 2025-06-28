from typing import Set

# virtual boltzmann constant. Not to be used in real calculations, but simply for chemical loss calculations
KB: float = 1.0 

# AMBER caps are used to represent the N-terminus and C-terminus of a protein
AMBER_CAPS: Set[str] = {"ACE", "NME", "NHE"}

CANONICAL_AMINO_ACID_3_LETTER_CODES: Set[str] = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 
    'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 
    'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}

OTHER_ALLOWED_RESIDUES: Set[str] = set() # {'ORN'} # we will support ornithine later

ALLOWED_RESIDUES: Set[str] = CANONICAL_AMINO_ACID_3_LETTER_CODES | AMBER_CAPS | OTHER_ALLOWED_RESIDUES