from abc import ABCMeta
from typing import Optional, Dict, Sequence, List, TypeVar

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from ..utils.constants import AMBER_CAPS, CANONICAL_AMINO_ACID_3_LETTER_CODES
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS

# Type variable for generic return types
T = TypeVar('T', bound='CysteineCarbo')

class CysteineCarbo(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for cysteine-carboxyl chemical interactions in protein structures.
    
    Attributes:
        _atom_idxs_keys (Tuple[str, ...]): The four atoms defining the tetrahedral geometry:
            - 'S1': Sulfur atom of the cysteine (SG)
            - 'C1': CB carbon behind the sulfur in cysteine
            - 'C3': Central carbon of the carboxyl group
            - 'O1': Oxygen atom of the carboxyl group
        _kde_file (str): Path to the KDE model file (.pt) for the statistical potential
    """
    _atom_idxs_keys = (
        'S1',  # sulfur of the cysteine
        'C1',  # the carbon behind the sulfur
        'C3',  # central carbon of the carboxyl
        'O1',  # an oxygen of the carboxyl (potentially resonant)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['cysteine-carbo']

class CysAsp(CysteineCarbo):
    """
    Cysteine's carboxyl group forms a bond with an aspartic acid carboxyl group.
    """
    _method = "CysAsp"
    
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['CysAsp', ...]:
        
        donor_atom_groups = {
            'S1': ['SG'],        # Cysteine's sulfur
            'C1': ['CB'],        # Cysteine's carbon behind the sulfur
        }
        
        acceptor_atom_groups = {
            'C3': ['CG'],        # Aspartic acid's carboxyl carbon
            'O1': ['OD1', 'OD2'], # Aspartic acid's carboxyl oxygens (resonant forms)
        }
        
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='CYS',
            acceptor_resname='ASP',
            donor_atom_groups=donor_atom_groups,
            acceptor_atom_groups=acceptor_atom_groups,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )
    
class CysGlu(CysteineCarbo):
    """
    Cysteine's carboxyl group forms a bond with a glutamic acid carboxyl group.
    """
    _method = "CysGlu"
    
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['CysGlu', ...]:
        donor_atom_groups = {
            'S1': ['SG'],        # Cysteine's sulfur
            'C1': ['CB'],        # Cysteine's carbon behind the sulfur
        }
        acceptor_atom_groups = {
            'C3': ['CD'],        # Glutamic acid's carboxyl carbon
            'O1': ['OE1', 'OE2'], # Glutamic acid's carboxyl oxygens (resonant forms)
        }
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='CYS',
            acceptor_resname='GLU',
            donor_atom_groups=donor_atom_groups,
            acceptor_atom_groups=acceptor_atom_groups,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )

class CysCTerm(CysteineCarbo):
    """
    Cysteine's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = "CysCTerm"
    _required_to_tail_keys = ('S1', 'C1')  # the keys that must be present in the res_atom_name_dict
    _valid_oxygen_names: tuple[str, ...] = ('O', 'OXT')

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['CysCTerm', ...]:
        return cls._res_to_tail(
            traj=traj,
            res_name='CYS',
            res_atom_name_dict={'S1': ['SG'], 'C1': ['CB']},
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )

    @classmethod
    def _validate_res_to_tail_inputs(cls, 
                                    traj: md.Trajectory, 
                                    res_name: str, 
                                    res_atom_name_dict: Dict[str, Sequence[str]],
                                    ) -> None:
        if res_name not in CANONICAL_AMINO_ACID_3_LETTER_CODES:
            raise ValueError(f"""Invalid res_name: {res_name}. Must be a canonical amino acid 3 letter code.
                             Valid codes are: {CANONICAL_AMINO_ACID_3_LETTER_CODES}""")
        
        if set(res_atom_name_dict.keys()) != set(cls._required_to_tail_keys):
            raise ValueError(f"""res_atom_name_key_dict must have keys of exactly {cls._required_to_tail_keys}.
                             Got {tuple(res_atom_name_dict.keys())}""")
        
        # Get all residues, excluding common caps
        exclude_residue_names = AMBER_CAPS
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
    def _res_to_tail(cls: type[T], 
                     traj: md.Trajectory, 
                     res_name: str, 
                     res_atom_name_dict: Dict[str, Sequence[str]],
                     atom_indexes_dict: AtomIndexDict,
                     weight: float = 1.0,
                     offset: float = 0.0,
                     temp: float = 1.0,
                     device: Optional[torch.device] = None,
                     ) -> tuple[T, ...]:
        """
        Helper method to make `get_loss_instances` methods easier to write for losses that connect
        a sidechain to a terminal group via a carboxyl bond. Returns a tuple of instances of the calling class.
        """
        cls._validate_res_to_tail_inputs(traj, res_name, res_atom_name_dict)

        valid_residues = [res for res in traj.topology.residues if res.name not in AMBER_CAPS]
        c_term_residue = valid_residues[-1]

        c_term_atom_groups = {
                'C3': ['C'],                          # C-terminal carboxyl carbon
                'O1': list(cls._valid_oxygen_names),  # C-terminal carboxyl oxygens
            }
        c_term_atoms = cls._find_valid_atoms_for_residue(
                c_term_residue.index, c_term_atom_groups, atom_indexes_dict
            )
        if not c_term_atoms:
                return ()
        
        user_specified_residue_list = [res for res in valid_residues if res.name == res_name]
        
        # Exclude the C-terminal residue from user-specified residues to prevent self-bonding
        # when the target residue type is the same as the C-terminal residue
        user_specified_residue_list = [res for res in user_specified_residue_list if res.index != c_term_residue.index]
        if len(user_specified_residue_list) == 0:
            return ()
        
        losses: List[T] = []
        for user_specified_residue in user_specified_residue_list:
            user_res_atoms = cls._find_valid_atoms_for_residue(
                    user_specified_residue.index, res_atom_name_dict, atom_indexes_dict
                )
            if not user_res_atoms:
                continue
            
            # Combine terminal and user residue atoms
            available_atoms = {**c_term_atoms, **user_res_atoms}
            resonance_key = (cls._method, frozenset([c_term_residue.index, user_specified_residue.index]))
                
            # Generate loss instances for all atom combinations
            instances = cls._generate_loss_instances_for_pair(
                available_atoms, resonance_key, weight, offset, temp, device
            )
            losses.extend(instances)
        
        return tuple(losses)