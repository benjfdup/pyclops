from typing import Dict, List
from abc import ABCMeta

import mdtraj as md

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from ..utils.utils import inherit_docstring
from ..losses.standard_kde_locations import STANDARD_KDE_LOCATIONS


class Amide(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for all amide bond cyclization chemistries.
    
    An amide bond forms between a nitrogen (usually from an amine group)
    and a carbon from a carboxyl group, with the loss of water.
    
    The geometry is defined by four atoms:
    - N1: The nitrogen involved in the amide bond
    - C1: The carbon of the carboxyl group
    - O1: An oxygen from the carboxyl group
    - C2: The carbon 'behind' the nitrogen (in its amino acid)
    """
    atom_idxs_keys = [
        'N1',  # The nitrogen directly involved in the bond
        'C1',  # Carbon of the carboxyl group
        'O1',  # Oxygen of the carboxyl group
        'C2',  # Carbon 'behind' the nitrogen (in its same amino acid)
    ]
    kde_file = STANDARD_KDE_LOCATIONS['amide'] # Statistical potential for amide bond geometry


class AmideHead2Tail(Amide):
    """
    Backbone-to-backbone amide bond between the N-terminus and C-terminus.
    
    This represents a standard head-to-tail cyclization where the C-terminal
    carboxyl group forms an amide bond with the N-terminal amine.
    """
    @classmethod
    @inherit_docstring(Amide.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names=["*"],  # Any residue can be a donor (C-terminal)
            acceptor_residue_names=["*"],  # Any residue can be an acceptor (N-terminal)
            donor_atom_groups={
                'C1': ['C'],        # C-terminal carboxyl carbon
                'O1': ['O', 'OXT'],  # C-terminal carboxyl oxygens (resonant)
            },
            acceptor_atom_groups={
                'N1': ['N'],        # N-terminal nitrogen
                'C2': ['CA'],       # Alpha carbon of N-terminal residue
            },
            method_name="AmideHead2Tail",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect N-term to C-term
            special_selection=lambda donors, acceptors: [
                (donors[-1], acceptors[0])  # Last residue to first residue
            ]
        )


class AmideSide2Side(Amide):
    """
    Base class for sidechain-to-sidechain amide bond cyclization.
    
    This represents amide bonds formed between sidechains, typically
    involving lysine's amine group and aspartate/glutamate's carboxyl group.
    """
    pass


class AmideLysGlu(AmideSide2Side):
    """
    Amide bond between Lysine's sidechain NH3+ and Glutamate's sidechain COO-.
    
    This represents a specific sidechain-to-sidechain cyclization where
    lysine's terminal amine forms an amide bond with glutamate's carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Side.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names="GLU",
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine sidechain nitrogen
                'C2': ['CE'],  # Carbon behind lysine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['CD'],              # Glutamate central carbon
                'O1': ['OE1', 'OE2'],      # Carboxyl oxygens (resonant)
            },
            method_name="AmideLysGlu",
            exclude_residue_names=["ACE", "NME", "NHE"]
        )


class AmideLysAsp(AmideSide2Side):
    """
    Amide bond between Lysine's sidechain NH3+ and Aspartate's sidechain COO-.
    
    This represents a specific sidechain-to-sidechain cyclization where
    lysine's terminal amine forms an amide bond with aspartate's carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Side.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names="ASP",
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine sidechain nitrogen
                'C2': ['CE'],  # Carbon behind lysine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['CG'],              # Aspartate central carbon
                'O1': ['OD1', 'OD2'],      # Carboxyl oxygens (resonant)
            },
            method_name="AmideLysAsp",
            exclude_residue_names=["ACE", "NME", "NHE"]
        )


class AmideOrnGlu(AmideSide2Side):
    """
    Amide bond between Ornithine's sidechain NH3+ and Glutamate's sidechain COO-.
    
    This represents a specific sidechain-to-sidechain cyclization where
    ornithine's terminal amine forms an amide bond with glutamate's carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Side.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="ORN",
            acceptor_residue_names="GLU",
            donor_atom_groups={
                'N1': ['NZ'],  # Ornithine sidechain nitrogen
                'C2': ['CD'],  # Carbon behind ornithine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['CD'],              # Glutamate central carbon
                'O1': ['OE1', 'OE2'],      # Carboxyl oxygens (resonant)
            },
            method_name="AmideOrnGlu",
            exclude_residue_names=["ACE", "NME", "NHE"]
        )


class AmideSide2Head(Amide):
    """
    Base class for sidechain-to-N-terminal amide bond cyclization.
    
    This represents amide bonds formed between a sidechain amine group
    and the N-terminal carboxyl group.
    """
    pass


class AmideLysHead(AmideSide2Head):
    """
    Amide bond between Lysine's sidechain NH3+ and the N-terminal carboxyl.
    
    This represents a specific sidechain-to-terminus cyclization where
    lysine's terminal amine forms an amide bond with the N-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Head.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names=["*"],  # Any residue can be the N-terminal
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine sidechain nitrogen
                'C2': ['CE'],  # Carbon behind lysine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['C'],               # N-terminal carboxyl carbon
                'O1': ['O', 'OXT'],        # Terminal oxygens (resonant)
            },
            method_name="AmideLysHead",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the N-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[0]) for donor in donors if donor.index != acceptors[0].index
            ]
        )


class AmideArgHead(AmideSide2Head):
    """
    Amide bond between Arginine's sidechain guanidino group and the N-terminal carboxyl.
    
    This represents a specific sidechain-to-terminus cyclization where
    arginine's guanidino nitrogens form an amide bond with the N-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Head.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="ARG",
            acceptor_residue_names=["*"],  # Any residue can be the N-terminal
            donor_atom_groups={
                'N1': ['NH1', 'NH2'],  # Arginine guanidino nitrogens (resonant)
                'C2': ['CZ'],          # Carbon connected to the guanidino group
            },
            acceptor_atom_groups={
                'C1': ['C'],           # N-terminal carboxyl carbon
                'O1': ['O', 'OXT'],    # Terminal oxygens (resonant)
            },
            method_name="AmideArgHead",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the N-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[0]) for donor in donors if donor.index != acceptors[0].index
            ]
        )


class AmideSide2Tail(Amide):
    """
    Base class for sidechain-to-C-terminal amide bond cyclization.
    
    This represents amide bonds formed between a sidechain amine group
    and the C-terminal carboxyl group.
    """
    pass


class AmideLysTail(AmideSide2Tail):
    """
    Amide bond between Lysine's sidechain NH3+ and the C-terminal carboxyl.
    
    This represents a specific sidechain-to-terminus cyclization where
    lysine's terminal amine forms an amide bond with the C-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Tail.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names=["*"],  # Any residue can be the C-terminal
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine sidechain nitrogen
                'C2': ['CE'],  # Carbon behind lysine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['C'],               # C-terminal carboxyl carbon
                'O1': ['O', 'OXT'],        # Terminal oxygens (resonant)
            },
            method_name="AmideLysTail",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the C-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[-1]) for donor in donors if donor.index != acceptors[-1].index
            ]
        )


class AmideOrnTail(AmideSide2Tail):
    """
    Amide bond between Ornithine's sidechain NH3+ and the C-terminal carboxyl.
    
    This represents a specific sidechain-to-terminus cyclization where
    ornithine's terminal amine forms an amide bond with the C-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(AmideSide2Tail.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="ORN",
            acceptor_residue_names=["*"],  # Any residue can be the C-terminal
            donor_atom_groups={
                'N1': ['NZ'],  # Ornithine sidechain nitrogen
                'C2': ['CD'],  # Carbon behind ornithine nitrogen
            },
            acceptor_atom_groups={
                'C1': ['C'],               # C-terminal carboxyl carbon
                'O1': ['O', 'OXT'],        # Terminal oxygens (resonant)
            },
            method_name="AmideOrnTail",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the C-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[-1]) for donor in donors if donor.index != acceptors[-1].index
            ]
        )