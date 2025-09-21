from typing import Optional, Tuple, final, Dict, Sequence, List

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from ..utils.constants import AMBER_CAPS
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS
from ..utils.constants import CANONICAL_AMINO_ACID_3_LETTER_CODES


class NitPhe(ChemicalLoss):
    """
    Class for nitrogen groups interacting with phenylalanine ring.
    """
    _atom_idxs_keys = (  # DO NOT CHANGE THE ORDER OF THESE KEYS, WILL AFFECT THE KDE CALCULATION & BREAK THE CODE
        'N1',  # AA1 Nitrogen atom
        'C1',  # AA1 carbon behind the nitrogen atom (sometimes the alpha carbon, sometimes a sidechain carbon)

        'C3',  # AA2 Furthest carbon of the phenylalanine ring (CZ)
        'C2',  # AA2 CE1 or CE2 of the phenylalanine ring (this is potentially resonant)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['nit-phe']
    _common_caps = AMBER_CAPS
    _res_to_term_terminal_names = ('head',) # names of allowed terminal specifications for the `_res_to_term` method
    _required_to_head_keys = ('C3', 'C2') # Since we need a Phenylalanine ring carbon and a Nitrogen to bond with

    @classmethod
    def _validate_res_to_term_inputs(cls,
                                     traj: md.Trajectory,
                                     res_name: str,
                                     res_atom_name_dict: Dict[str, Sequence[str]],
                                     term_name: str,
                                     ) -> None:
        """
        Validate the inputs for the `_res_to_term` method.
        """
        if term_name not in cls._res_to_term_terminal_names:
            raise ValueError(f"Invalid term name: {term_name}. Must be one of {cls._res_to_term_terminal_names}")
        if term_name == 'head':
            if set(res_atom_name_dict.keys()) != set(cls._required_to_head_keys):
                raise ValueError(f"""Invalid keys in res_atom_name_dict for 'head' as term_name: 
                                 {tuple(res_atom_name_dict.keys())}. Must be {cls._required_to_head_keys}""")
        if res_name not in CANONICAL_AMINO_ACID_3_LETTER_CODES:
            raise ValueError(f"""Invalid res_name: {res_name}. Must be a canonical amino acid 3 letter code.
                             Valid codes are: {CANONICAL_AMINO_ACID_3_LETTER_CODES}""")
        
        # Get all residues, excluding common caps
        exclude_residue_names = cls._common_caps
        all_residues = list(traj.topology.residues)
        
        # Check that excluded residue names only appear in terminal positions
        for i, residue in enumerate(all_residues):
            if residue.name in exclude_residue_names:
                is_first = (i == 0)
                is_last = (i == len(all_residues) - 1)
                if not (is_first or is_last):
                    raise ValueError(
                        f"Common cap residue '{residue.name}' found at non-terminal position "
                        f"(residue {i+1} of {len(all_residues)}). Common caps should only "
                        f"appear at the N-terminus (first) or C-terminus (last) positions."
                    )

    @classmethod
    def _res_to_term(cls, 
                     traj: md.Trajectory, 
                     res_name: str, 
                     res_atom_name_dict: Dict[str, Sequence[str]],
                     term_name: str,
                     atom_indexes_dict: AtomIndexDict,
                     weight: float = 1.0,
                     offset: float = 0.0,
                     temp: float = 1.0,
                     device: Optional[torch.device] = None,
                     ) -> Tuple['NitPhe', ...]:
        """
        Helper method to make `get_loss_instances` methods easier to write for losses that connect
        a sidechain to a terminal group. Returns a tuple of `NitPhe` instances (though
        in reality this will be used by subclasses to return a tuple of `cls` instances).

        Args:
            traj: MDTraj trajectory for analysis and atom identification
            res_name: Name of the residue to connect to the terminal groups
            res_atom_name_dict: Dictionary mapping from _atom_idxs_keys members to possible atom names in the residue 
            that connects to the terminal group. If term_name is 'head', then the keys should be C3 and C2. 
            term_name: Name of the terminal group to connect to the residue (head or tail)
            atom_indexes_dict: Dictionary mapping (residue_idx, atom_name) to global atom indices (int)
            weight: Weight of the loss
            offset: Offset of the loss
            temp: Temperature of the loss
            device: Device to run the loss on
        """
        cls._validate_res_to_term_inputs(traj, res_name, res_atom_name_dict, term_name)
        
        # Get all residues, excluding common caps
        exclude_residue_names = cls._common_caps
        all_residues = list(traj.topology.residues)
        
        valid_residues = [r for r in all_residues if r.name not in exclude_residue_names]
        if len(valid_residues) < 1:
            return ()
        
        losses: List['NitPhe'] = []
        user_specified_residue_list: List[md.Residue] = [r for r in valid_residues if r.name == res_name]
        if len(user_specified_residue_list) == 0:
            return ()
        
        if term_name == 'head':
            # N-terminal atoms - the head provides N1 and C2
            n_term_residue = valid_residues[0]
            n_term_atom_groups = {
                'N1': ['N'],      # N-terminal nitrogen
                'C1': ['CA'],     # N-terminal alpha carbon
            }
            
            # Find valid N-terminal atoms
            n_term_atoms = cls._find_valid_atoms_for_residue(
                n_term_residue.index, n_term_atom_groups, atom_indexes_dict
            )
            if not n_term_atoms:
                return ()
            
            # Process each user-specified residue for C1 and O1
            for user_specified_residue in user_specified_residue_list:
                # Find valid atoms for the user-specified residue
                user_res_atoms = cls._find_valid_atoms_for_residue(
                    user_specified_residue.index, res_atom_name_dict, atom_indexes_dict
                )
                if not user_res_atoms:
                    continue
                
                # Combine terminal and user residue atoms
                available_atoms = {**n_term_atoms, **user_res_atoms}
                resonance_key = (cls._method, frozenset([n_term_residue.index, user_specified_residue.index]))
                
                # Generate loss instances for all atom combinations
                instances = cls._generate_loss_instances_for_pair(
                    available_atoms, resonance_key, weight, offset, temp, device
                )
                losses.extend(instances)
        else:
            raise ValueError(f"Invalid term_name: {term_name}. Must be one of {cls._res_to_term_terminal_names}")
        
        return tuple(losses)

# sidechain to head
@final
class PheHead(NitPhe):
    """
    Class for phenylalanine-head-amide chemical interactions.
    """
    _method = "PheHead"
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> Tuple['PheHead', ...]:
        
        return cls._res_to_term(
            traj=traj,
            res_name='PHE',
            res_atom_name_dict={'C3': ['CZ'], 'C2': ['CE2', 'CEy', 'CE%', 'CE1', 'CEx']},
            term_name='head',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )

# NitPheSide2Side
class NitPheSide2Side(NitPhe):
    """
    Base class for sidechain-to-sidechain nit-phe bond cyclization.
    """
    pass

@final
class LysPhe(NitPheSide2Side):
    """
    Class for lysine-phenylalanine chemical interactions.
    """
    _method = "LysPhe"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> Tuple['LysPhe', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='PHE',
            donor_atom_groups={'N1': ['NZ'], 'C1': ['CD']},
            acceptor_atom_groups={'C3': ['CZ'], 'C2': [
                'CE1', 'CE2', 'CEx', 'CEy', 'CE%']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )
