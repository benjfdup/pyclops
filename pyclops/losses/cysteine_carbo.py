from abc import ABCMeta
from typing import Dict, List
import warnings

import mdtraj as md

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from .standard_file_locations import STANDARD_KDE_LOCATIONS, STANDARD_LINKAGE_PDB_LOCATIONS


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
    linkage_pdb_file = STANDARD_LINKAGE_PDB_LOCATIONS['cysteine-carbo']

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
    
    @staticmethod
    def _get_cterm_pairings( # review this method...
        traj: md.Trajectory,
        atom_indexes_dict: Dict,
        terminal_o_variants: List[str] = ("O", "OXT")  # Allow both for resonance, prioritize double bond
        ) -> List[IndexesMethodPair]:
        """
        Finds all valid cysteine → C-terminal pairings for carboxyl loss chemistry,
        considering resonance between carboxylate oxygens.

        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing residue and atom information.

        atom_indexes_dict : Dict[Tuple[int, str], int]
            Maps (residue index, atom name) to atom indices.

        terminal_o_variants : List[str]
            Variants for the terminal oxygen atom names. Defaults to ("O", "OXT").

        Returns
        -------
        List[IndexesMethodPair]
        """
        pairs = []
        residues = list(traj.topology.residues)
        cys_residues = [r for r in residues if r.name == "CYS"]
        real_residues = [r for r in residues if r.name not in {"NME", "NHE"}]

        if len(real_residues) < 2:
            warnings.warn("[CysCTerm] Not enough residues to define C-terminal pairing.")
            return pairs

        cterm = real_residues[-1]

        # Accept only if both oxygens exist, since we're assuming resonance equivalence
        found_oxygens = [name for name in terminal_o_variants if (cterm.index, name) in atom_indexes_dict]

        if len(found_oxygens) < 2:
            warnings.warn(f"[CysCTerm] Skipping C-terminal residue {cterm.name} {cterm.index}: "
                        f"found fewer than 2 resonance oxygens.")
            return pairs

        for cys in cys_residues:
            for o_name in found_oxygens:
                try:
                    s1 = atom_indexes_dict[(cys.index, "SG")]
                    c1 = atom_indexes_dict[(cys.index, "CB")]
                    c3 = atom_indexes_dict[(cterm.index, "C")]
                    o1 = atom_indexes_dict[(cterm.index, o_name)]

                    atom_dict = {
                        "S1": s1,
                        "C1": c1,
                        "C3": c3,
                        "O1": o1
                    }

                    method_str = f"CysCTerm, CYS {cys.index} -> C-term {cterm.name} {cterm.index} (resonant: {o_name})"
                    pairs.append(IndexesMethodPair(atom_dict, method_str, {cys.index, cterm.index}))

                except KeyError as e:
                    warnings.warn(f"[CysCTerm] Missing atom: {e}")
                    continue

        return pairs

    @classmethod
    #@inherit_docstring(CysCarboxyl.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls._get_cterm_pairings(traj, atom_indexes_dict)

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