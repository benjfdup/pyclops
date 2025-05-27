from typing import Dict, List

import mdtraj as md
import torch

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from ..utils.utils import inherit_docstring
from ..losses.standard_kde_locations import STANDARD_KDE_LOCATIONS


class LysArg(ChemicalLoss):
    """
    Models the cyclization chemistry between lysine and arginine side chains.
    
    This class represents a specific cyclization where a lysine's terminal amine
    interacts with the guanidino group of an arginine side chain. The geometry is
    defined by four nitrogen atoms:
    
    - N1: The terminal nitrogen of the lysine side chain (NZ)
    - N2 & N3: The outer nitrogens of the arginine guanidino group (NH1 & NH2)
    - N4: The inner nitrogen of the arginine guanidino group (NE)
    
    The loss function evaluates how favorable this geometry is for potential
    cyclization, based on a statistical potential derived from empirical data.
    """
    
    atom_idxs_keys = [
        'N1',  # Nitrogen of the Lysine (NZ)
        'N2',  # One outer nitrogen of the arginine (NH1/NH2, resonant with N3)
        'N3',  # The other outer nitrogen of the arginine (NH2/NH1, resonant with N2)
        'N4',  # The "inner" nitrogen of the arginine (NE)
    ]
    
    kde_file = STANDARD_KDE_LOCATIONS['lys-arg']
    
    @classmethod
    @inherit_docstring(ChemicalLoss.get_indexes_and_methods)
    def get_indexes_and_methods(cls, traj: md.Trajectory, atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid lysine-arginine pairings for potential cyclization.
        
        This method identifies all possible interactions between lysine and arginine
        side chains, considering resonance structures in the guanidino group of arginine.
        """
        pairs = cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="LYS",
            acceptor_residue_names="ARG",
            donor_atom_groups={
                'N1': ['NZ'],  # Lysine terminal nitrogen
            },
            acceptor_atom_groups={
                'N2': ['NH1', 'NH2'],  # Outer nitrogens (resonant)
                'N3': ['NH2', 'NH1'],  # Outer nitrogens (resonant, opposite to N2)
                'N4': ['NE'],          # Inner nitrogen
            },
            method_name="LysArg",
            exclude_residue_names=["ACE", "NME", "NHE"]
        )
        
        # Post-process the pairs to ensure chemical consistency
        return cls.post_process_pairs(pairs, atom_indexes_dict)
    
    @classmethod
    def post_process_pairs(cls, pairs: List[IndexesMethodPair], atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Post-processes the pairs found by find_valid_pairs to ensure chemical consistency.
        
        This method ensures that:
        1. When NH1 is selected for N2, NH2 is selected for N3 and vice versa
        2. Each resonance pair is properly labeled in the method string
        
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
        
        for pair in pairs:
            # Extract the residue indices
            residue_indices = list(pair.pair)
            
            # Determine which resonance is being used
            # Find which residue is ARG (should be the one with NH1/NH2)
            # First, parse the method string to find residue types and indices
            method_parts = pair.method.split(' (resonant:', 1)[0]  # Remove any existing resonance info
            method_parts = method_parts.split(', ')[1].split(' -> ')
            
            if method_parts[0].startswith('LYS') and method_parts[1].startswith('ARG'):
                lys_res_idx = int(method_parts[0].split(' ')[1])
                arg_res_idx = int(method_parts[1].split(' ')[1])
                
                # Get the atom indices for resonant atoms
                n2_idx = pair.indexes['N2']
                n3_idx = pair.indexes['N3']
                
                # Check if the resonance is consistent
                # If N2 is NH1, then N3 should be NH2 and vice versa
                nh1_idx = atom_indexes_dict.get((arg_res_idx, 'NH1'))
                nh2_idx = atom_indexes_dict.get((arg_res_idx, 'NH2'))
                
                if n2_idx == nh1_idx and n3_idx == nh2_idx:
                    resonance_str = "NH1-NH2"
                elif n2_idx == nh2_idx and n3_idx == nh1_idx:
                    resonance_str = "NH2-NH1"
                else:
                    # Skip invalid resonance pairs
                    continue
                
                # Update the method string with our specific resonance format
                updated_method = f"LysArg, LYS {lys_res_idx} -> ARG {arg_res_idx} (resonant: {resonance_str})"
                
                # Create a new pair with the updated method string
                valid_pair = IndexesMethodPair(
                    pair.indexes.copy(),
                    updated_method,
                    pair.pair
                )
                
                valid_pairs.append(valid_pair)
                    
        return valid_pairs