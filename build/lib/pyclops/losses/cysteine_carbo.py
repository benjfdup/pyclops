from abc import ABCMeta
from typing import Dict, List

import mdtraj as md

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from ..losses.standard_kde_locations import STANDARD_KDE_LOCATIONS


class CysCarboxyl(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for cysteine-carboxyl chemical interactions in protein structures.
    
    This abstract class models the interaction between a cysteine thiol group 
    and a carboxyl group, which is a common reactive chemistry in proteins.
    The interaction is modeled using a tetrahedral geometry represented by four
    key atoms and evaluated using a KDE-based statistical potential. Based on the 
    chemistry described in Bechtler & Lamers, 2021.
    
    Subclasses implement specific types of carboxyl interactions (C-terminal,
    aspartic acid, glutamic acid) by defining their atom selection strategy.
    
    Attributes:
        atom_idxs_keys (List[str]): The four atoms defining the tetrahedral geometry:
            - 'S1': Sulfur atom of the cysteine (SG)
            - 'C3': Central carbon of the carboxyl group
            - 'O1': Oxygen atom of the carboxyl group
            - 'C1': CB carbon behind the sulfur in cysteine
        kde_file (str): Path to the KDE model file (.pt) for the statistical potential
    
    Notes:
        This is an abstract base class and should not be instantiated directly.
        Use one of the concrete subclasses (CysCTerm, CysAsp, CysGlu) instead.
    """
    
    atom_idxs_keys = [
        'S1',  # sulfur of the cysteine
        'C3',  # central carbon of the carboxyl
        'O1',  # oxygen of the carboxyl
        'C1',  # carbon behind the sulfur
    ]
    kde_file = STANDARD_KDE_LOCATIONS['cysteine-carbo']

class CysCTerm(CysCarboxyl):
    """
    Models the potential chemical interaction between a cysteine thiol and a C-terminal carboxyl group.
    
    This loss function guides the geometry of a cysteine side chain and a protein 
    C-terminus to form configurations favorable for nucleophilic attack of the 
    cysteine thiol on the C-terminal carboxyl carbon, considering the resonance 
    between terminal oxygens (O and OXT).
    
    Notes:
        - Only considers the final residue of the protein chain as the C-terminus
        - Takes into account resonance structures where either terminal oxygen
          can be the primary acceptor
    """
    
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid cysteine → C-terminal pairings for carboxyl loss chemistry.
        
        This method identifies all possible interactions between cysteine thiols
        and the C-terminal carboxylate group, considering resonance structures.
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing residue and atom information.
        
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_index, atom_name) to atom index.
            
        Returns
        -------
        List[IndexesMethodPair]
            List of valid index-method pairs for all CysCTerm interactions.
            Each pair contains:
            - Dictionary mapping from atom keys to atom indices
            - Method string describing the specific interaction
            - Set of involved residue indices
            
        Notes
        -----
        Uses the find_valid_pairs utility with terminal-specific selection to
        ensure only the true C-terminus is considered as an acceptor.
        """
        
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="CYS",
            acceptor_residue_names=None,  # Will use special selection for C-terminal
            donor_atom_groups={
                'S1': ['SG'],
                'C1': ['CB']
            },
            acceptor_atom_groups={
                'C3': ['C'],
                'O1': ['O', 'OXT']  # Resonance between these oxygens
            },
            method_name="CysCTerm",
            require_terminals=True,
            special_selection=lambda donors, _: [
                # Select only C-terminal residue as acceptor
                (donor, list(traj.topology.residues)[-1]) 
                for donor in donors 
                if donor.index != list(traj.topology.residues)[-1].index
            ]
        )

class CysAsp(CysCarboxyl):
    """
    Models the potential chemical interaction between a cysteine thiol and an aspartic acid side chain.
    
    This loss function guides the geometry of cysteine and aspartic acid side chains
    to form configurations favorable for nucleophilic attack of the cysteine thiol
    on the aspartic acid carboxyl carbon, considering resonance between the
    carboxylate oxygens (OD1 and OD2).
    
    This interaction can represent various protein chemistries including thioester
    formation and related acyl transfer reactions.
    """
    
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid cysteine → aspartic acid pairings for carboxyl loss chemistry.
        
        This method identifies all possible interactions between cysteine thiols
        and aspartic acid carboxylate groups, considering resonance structures.
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing residue and atom information.
        
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_index, atom_name) to atom index.
            
        Returns
        -------
        List[IndexesMethodPair]
            List of valid index-method pairs for all CysAsp interactions.
            Each pair contains:
            - Dictionary mapping from atom keys to atom indices
            - Method string describing the specific interaction
            - Set of involved residue indices
            
        Notes
        -----
        Automatically handles resonance between OD1 and OD2 atoms of aspartic acid,
        generating appropriate configurations for all possible resonance structures.
        """
        
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="CYS",
            acceptor_residue_names="ASP",
            donor_atom_groups={
                'S1': ['SG'],
                'C1': ['CB']
            },
            acceptor_atom_groups={
                'C3': ['CG'],
                'O1': ['OD1', 'OD2']  # Resonance between these oxygens
            },
            method_name="CysAsp"
        )

class CysGlu(CysCarboxyl):
    """
    Models the potential chemical interaction between a cysteine thiol and a glutamic acid side chain.
    
    This loss function guides the geometry of cysteine and glutamic acid side chains
    to form configurations favorable for nucleophilic attack of the cysteine thiol
    on the glutamic acid carboxyl carbon, considering resonance between the
    carboxylate oxygens (OE1 and OE2).
    
    Similar to CysAsp, but represents interactions with the longer glutamic acid
    side chain, which may offer different geometrical constraints due to the additional
    methylene group.
    """
    
    @classmethod
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid cysteine → glutamic acid pairings for carboxyl loss chemistry.
        
        This method identifies all possible interactions between cysteine thiols
        and glutamic acid carboxylate groups, considering resonance structures.
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing residue and atom information.
        
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_index, atom_name) to atom index.
            
        Returns
        -------
        List[IndexesMethodPair]
            List of valid index-method pairs for all CysGlu interactions.
            Each pair contains:
            - Dictionary mapping from atom keys to atom indices
            - Method string describing the specific interaction
            - Set of involved residue indices
            
        Notes
        -----
        Automatically handles resonance between OE1 and OE2 atoms of glutamic acid,
        generating appropriate configurations for all possible resonance structures.
        """
        
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="CYS",
            acceptor_residue_names="GLU",
            donor_atom_groups={
                'S1': ['SG'],
                'C1': ['CB']
            },
            acceptor_atom_groups={
                'C3': ['CD'],
                'O1': ['OE1', 'OE2']  # Resonance between these oxygens
            },
            method_name="CysGlu"
        )