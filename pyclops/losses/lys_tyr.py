from typing import Dict, List

import mdtraj as md

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from ..utils.utils import inherit_docstring
from .standard_file_locations import STANDARD_KDE_LOCATIONS


class LysTyr(ChemicalLoss):
    """
    Models the cyclization chemistry between lysine and tyrosine side chains.
    
    This class represents a specific cyclization where a lysine's terminal amine
    interacts with the hydroxyl group of a tyrosine side chain. The geometry is
    defined by four atoms:
    
    - N1: The terminal nitrogen of the lysine side chain (NZ)
    - C1: The delta carbon of the lysine side chain (CD)
    - O1: The hydroxyl oxygen of the tyrosine (OH)
    - C2: The zeta carbon of the tyrosine ring (CZ)
    
    The loss function evaluates how favorable this geometry is for potential
    cyclization, based on a statistical potential derived from empirical data.
    """
    
    atom_idxs_keys = [
        'N1',  # Nitrogen of the lysine (NZ)
        'C1',  # Delta carbon of the lysine (CD)
        'O1',  # Hydroxyl oxygen of the tyrosine (OH)
        'C2',  # Zeta carbon of the tyrosine ring (CZ)
    ]
    
    kde_file = STANDARD_KDE_LOCATIONS['lys-tyr']
    
    @classmethod
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid lysine-tyrosine pairings for potential cyclization.
        
        This method identifies all possible interactions between lysine terminal 
        amine groups and tyrosine hydroxyl groups that could potentially form
        a cyclization.
        """
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names="TYR",
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine terminal nitrogen
                'C1': ['CD'],  # Delta carbon of lysine
            },
            acceptor_atom_groups={
                'O1': ['OH'],  # Tyrosine hydroxyl oxygen
                'C2': ['CZ'],  # Zeta carbon of tyrosine ring
            },
            method_name="LysTyr",
            exclude_residue_names=["ACE", "NME", "NHE"]
        )