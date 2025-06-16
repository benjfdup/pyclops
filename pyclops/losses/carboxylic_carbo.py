from typing import Dict, List
from abc import ABCMeta

import mdtraj as md

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from ..utils.utils import inherit_docstring
from .standard_file_locations import STANDARD_KDE_LOCATIONS


class CarboxylicCarbo(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries.
    
    This class represents a family of cyclizations where two carboxyl groups
    react to form a cyclic anhydride. Based on the chemistry described in
    Bechtler & Lamers, 2021.
    
    The geometry is defined by four atoms:
    - O1: The non-double-bonded oxygen of the first carboxyl group
    - O2: The non-double-bonded oxygen of the second carboxyl group
    - C1: The carbon attached to the first oxygen
    - C2: The carbon attached to the second oxygen
    """
    atom_idxs_keys = [
        'O1',  # The non-double-bonded oxygen of the first carboxyl group
        'O2',  # The non-double-bonded oxygen of the second carboxyl group
        'C1',  # The carbon attached to the first oxygen
        'C2',  # The carbon attached to the second oxygen
    ]
    # Note: This path should be set properly in your actual implementation
    kde_file = STANDARD_KDE_LOCATIONS['carboxylic-carbo']

class AspGlu(CarboxylicCarbo):
    """
    Carboxyl-to-carboxyl cyclization between aspartate and glutamate sidechains.
    
    This represents a specific cyclization where the carboxyl groups of
    aspartate and glutamate residues react to form a cyclic anhydride.
    """
    @classmethod
    @inherit_docstring(CarboxylicCarbo.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names=["ASP", "GLU"],
            acceptor_residue_names=["ASP", "GLU"],
            donor_atom_groups={
                'O1': ['OD1', 'OD2', 'OE1', 'OE2'],  # All possible carboxyl oxygens
                'C1': ['CG', 'CD'],                   # Corresponding carbon atoms
            },
            acceptor_atom_groups={
                'O2': ['OD1', 'OD2', 'OE1', 'OE2'],  # All possible carboxyl oxygens
                'C2': ['CG', 'CD'],                   # Corresponding carbon atoms
            },
            method_name="AspGlu",
            exclude_residue_names=["ACE", "NME", "NHE"],
            # Custom selection to avoid self-pairing and duplicates
            special_selection=lambda donors, acceptors: [
                (donors[i], acceptors[j]) 
                for i in range(len(donors)) 
                for j in range(len(acceptors))
                if donors[i].index < acceptors[j].index  # Ensure unique pairings
                and donors[i].index != acceptors[j].index  # Avoid self-pairing
            ]
        )

    @classmethod
    def post_process_pairs(cls, pairs: List[IndexesMethodPair], atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Post-processes the pairs found by find_valid_pairs to ensure chemical consistency.
        
        This method ensures that:
        1. For ASP residues, we use CG as the carbon atom
        2. For GLU residues, we use CD as the carbon atom
        3. For ASP residues, we only use OD1/OD2 as oxygen atoms
        4. For GLU residues, we only use OE1/OE2 as oxygen atoms
        
        Parameters
        ----------
        pairs : List[IndexesMethodPair]
            The candidate pairs found by find_valid_pairs
            
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_idx, atom_name) to atom index
            
        Returns
        -------
        List[IndexesMethodPair]
            Filtered and corrected pairs
        """
        valid_pairs = []
        
        # Get residue information for lookup
        residue_info = {}
        for (res_idx, atom_name), atom_idx in atom_indexes_dict.items():
            if res_idx not in residue_info:
                # Find the residue name from any atom's entry
                for pair in pairs:
                    if res_idx in pair.pair:
                        # Infer residue type from method string
                        # Format is typically "AspGlu, ASP 5 -> GLU 10"
                        method_parts = pair.method.split(', ')[1].split(' -> ')
                        donor_info = method_parts[0].split(' ')
                        acceptor_info = method_parts[1].split(' ')
                        
                        if int(donor_info[1]) == res_idx:
                            residue_info[res_idx] = donor_info[0]
                            break
                        elif int(acceptor_info[1]) == res_idx:
                            residue_info[res_idx] = acceptor_info[0]
                            break
        
        # Now fix each pair
        for pair in pairs:
            # Extract residue indices from the pair
            res_indices = list(pair.pair)
            
            # Skip if we don't have residue info (should never happen)
            if not all(idx in residue_info for idx in res_indices):
                continue
                
            # Create a copy of the atom dict to modify
            new_atom_dict = pair.indexes.copy()
            
            # Fix the donor side (first residue)
            donor_res_idx = res_indices[0]
            donor_res_type = residue_info[donor_res_idx]
            
            # Fix the acceptor side (second residue)
            acceptor_res_idx = res_indices[1]
            acceptor_res_type = residue_info[acceptor_res_idx]
            
            # Update carbon atoms based on residue type
            if donor_res_type == "ASP":
                new_atom_dict["C1"] = atom_indexes_dict.get((donor_res_idx, "CG"))
            elif donor_res_type == "GLU":
                new_atom_dict["C1"] = atom_indexes_dict.get((donor_res_idx, "CD"))
                
            if acceptor_res_type == "ASP":
                new_atom_dict["C2"] = atom_indexes_dict.get((acceptor_res_idx, "CG"))
            elif acceptor_res_type == "GLU":
                new_atom_dict["C2"] = atom_indexes_dict.get((acceptor_res_idx, "CD"))
            
            # Ensure oxygen atoms are type-consistent
            # First, check if current O1 is valid for donor residue type
            current_o1 = new_atom_dict["O1"]
            valid_o1 = False
            
            if donor_res_type == "ASP":
                for o_name in ["OD1", "OD2"]:
                    if atom_indexes_dict.get((donor_res_idx, o_name)) == current_o1:
                        valid_o1 = True
                        break
                if not valid_o1:
                    # Try to find a valid oxygen
                    for o_name in ["OD1", "OD2"]:
                        if (donor_res_idx, o_name) in atom_indexes_dict:
                            new_atom_dict["O1"] = atom_indexes_dict[(donor_res_idx, o_name)]
                            valid_o1 = True
                            break
            elif donor_res_type == "GLU":
                for o_name in ["OE1", "OE2"]:
                    if atom_indexes_dict.get((donor_res_idx, o_name)) == current_o1:
                        valid_o1 = True
                        break
                if not valid_o1:
                    # Try to find a valid oxygen
                    for o_name in ["OE1", "OE2"]:
                        if (donor_res_idx, o_name) in atom_indexes_dict:
                            new_atom_dict["O1"] = atom_indexes_dict[(donor_res_idx, o_name)]
                            valid_o1 = True
                            break
            
            # Now check O2
            current_o2 = new_atom_dict["O2"]
            valid_o2 = False
            
            if acceptor_res_type == "ASP":
                for o_name in ["OD1", "OD2"]:
                    if atom_indexes_dict.get((acceptor_res_idx, o_name)) == current_o2:
                        valid_o2 = True
                        break
                if not valid_o2:
                    # Try to find a valid oxygen
                    for o_name in ["OD1", "OD2"]:
                        if (acceptor_res_idx, o_name) in atom_indexes_dict:
                            new_atom_dict["O2"] = atom_indexes_dict[(acceptor_res_idx, o_name)]
                            valid_o2 = True
                            break
            elif acceptor_res_type == "GLU":
                for o_name in ["OE1", "OE2"]:
                    if atom_indexes_dict.get((acceptor_res_idx, o_name)) == current_o2:
                        valid_o2 = True
                        break
                if not valid_o2:
                    # Try to find a valid oxygen
                    for o_name in ["OE1", "OE2"]:
                        if (acceptor_res_idx, o_name) in atom_indexes_dict:
                            new_atom_dict["O2"] = atom_indexes_dict[(acceptor_res_idx, o_name)]
                            valid_o2 = True
                            break
            
            # If both oxygens are valid, add this pair
            if valid_o1 and valid_o2:
                # Create a new pair with the corrected atom indices
                new_pair = IndexesMethodPair(
                    new_atom_dict,
                    pair.method,
                    pair.pair
                )
                valid_pairs.append(new_pair)
        
        return valid_pairs


class AspCTerm(CarboxylicCarbo):
    """
    Carboxyl-to-carboxyl cyclization between aspartate sidechain and C-terminal carboxyl.
    
    This represents a specific cyclization where the carboxyl group of an
    aspartate sidechain reacts with the C-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(CarboxylicCarbo.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="ASP",
            acceptor_residue_names=["*"],  # Any residue can be the C-terminal
            donor_atom_groups={
                'O1': ['OD1', 'OD2'],  # Aspartate carboxyl oxygens (resonant)
                'C1': ['CG'],          # Carbon attached to carboxyl group
            },
            acceptor_atom_groups={
                'O2': ['O', 'OXT'],   # C-terminal carboxyl oxygens (resonant)
                'C2': ['C'],          # C-terminal carbon
            },
            method_name="AspCTerm",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the C-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[-1]) for donor in donors if donor.index != acceptors[-1].index
            ]
        )


class GluCTerm(CarboxylicCarbo):
    """
    Carboxyl-to-carboxyl cyclization between glutamate sidechain and C-terminal carboxyl.
    
    This represents a specific cyclization where the carboxyl group of a
    glutamate sidechain reacts with the C-terminal carboxyl group.
    """
    @classmethod
    @inherit_docstring(CarboxylicCarbo.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="GLU",
            acceptor_residue_names=["*"],  # Any residue can be the C-terminal
            donor_atom_groups={
                'O1': ['OE1', 'OE2'],  # Glutamate carboxyl oxygens (resonant)
                'C1': ['CD'],          # Carbon attached to carboxyl group
            },
            acceptor_atom_groups={
                'O2': ['O', 'OXT'],   # C-terminal carboxyl oxygens (resonant)
                'C2': ['C'],          # C-terminal carbon
            },
            method_name="GluCTerm",
            exclude_residue_names=["ACE", "NME", "NHE"],
            require_terminals=True,
            # Custom selection to ensure we only connect to the C-terminal
            special_selection=lambda donors, acceptors: [
                (donor, acceptors[-1]) for donor in donors if donor.index != acceptors[-1].index
            ]
        )


# Additional carboxyl-to-carboxyl cyclizations can be added here
class AspAsp(CarboxylicCarbo):
    """
    Carboxyl-to-carboxyl cyclization between two aspartate sidechains.
    
    This represents a specific cyclization where the carboxyl groups of
    two aspartate residues react to form a cyclic anhydride.
    """
    @classmethod
    @inherit_docstring(CarboxylicCarbo.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="ASP",
            acceptor_residue_names="ASP",
            donor_atom_groups={
                'O1': ['OD1', 'OD2'],  # Aspartate carboxyl oxygens (resonant)
                'C1': ['CG'],          # Carbon attached to carboxyl group
            },
            acceptor_atom_groups={
                'O2': ['OD1', 'OD2'],  # Aspartate carboxyl oxygens (resonant)
                'C2': ['CG'],          # Carbon attached to carboxyl group
            },
            method_name="AspAsp",
            exclude_residue_names=["ACE", "NME", "NHE"],
            # Custom selection to avoid self-pairing and duplicates
            special_selection=lambda donors, acceptors: [
                (donors[i], acceptors[j]) 
                for i in range(len(donors)) 
                for j in range(i+1, len(acceptors))
            ]
        )


class GluGlu(CarboxylicCarbo):
    """
    Carboxyl-to-carboxyl cyclization between two glutamate sidechains.
    
    This represents a specific cyclization where the carboxyl groups of
    two glutamate residues react to form a cyclic anhydride.
    """
    @classmethod
    @inherit_docstring(CarboxylicCarbo.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="GLU",
            acceptor_residue_names="GLU",
            donor_atom_groups={
                'O1': ['OE1', 'OE2'],  # Glutamate carboxyl oxygens (resonant)
                'C1': ['CD'],          # Carbon attached to carboxyl group
            },
            acceptor_atom_groups={
                'O2': ['OE1', 'OE2'],  # Glutamate carboxyl oxygens (resonant)
                'C2': ['CD'],          # Carbon attached to carboxyl group
            },
            method_name="GluGlu",
            exclude_residue_names=["ACE", "NME", "NHE"],
            # Custom selection to avoid self-pairing and duplicates
            special_selection=lambda donors, acceptors: [
                (donors[i], acceptors[j]) 
                for i in range(len(donors)) 
                for j in range(i+1, len(acceptors))
            ]
        )