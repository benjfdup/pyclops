from abc import ABCMeta
from typing import Optional, Dict, Sequence, Tuple, List, TypeVar

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS
from ..utils.utils import _inherit_docstring
from ..utils.constants import CANONICAL_AMINO_ACID_3_LETTER_CODES, AMBER_CAPS

# Type variable for generic return types
T = TypeVar('T', bound='CarboxylicCarbo')

class CarboxylicCarbo(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries.
    
    This class represents a family of cyclizations where two carboxyl groups
    react to form a cyclic anhydride. Based on the chemistry described in
    Bechtler & Lamers, 2021.
    
    The geometry is defined by four atoms:
    - C1: The carbon from the first carboxyl group
    - O1: An oxygen from the first carboxyl group
    - C2: The carbon from the second carboxyl group
    - O2: An oxygen from the second carboxyl group
    """
    _atom_idxs_keys = ( # DO NOT CHANGE THE ORDER OF THESE KEYS, WILL AFFECT THE KDE CALCULATION & BREAK THE CODE
        'C1',  # the carbon of the first carboxyl group
        'O1',  # an oxygen of the first carboxyl group (potentially resonant)
        'C2',  # the carbon of the second carboxyl group
        'O2',  # an oxygen of the second carboxyl group (potentially resonant)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['carboxylic-carbo']

class Side2Side(CarboxylicCarbo):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where the
    carboxyl groups are each in a different amino acid sidechain.
    """
    pass

class AspGlu(Side2Side):
    """
    Aspartate's carboxyl group forms a bond with Glutamate's carboxyl group.
    """
    _method = "AspGlu"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AspGlu', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='ASP',
            acceptor_resname='GLU',
            donor_atom_groups={'C1': ['CG'], 'O1': ['OD1', 'OD2']},
            acceptor_atom_groups={'C2': ['CD'], 'O2': ['OE1', 'OE2']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )

class AspAsp(Side2Side):
    """
    Aspartate's carboxyl group forms a bond with itself.
    """
    _method = "AspAsp"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AspAsp', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='ASP',
            acceptor_resname='ASP',
            donor_atom_groups={'C1': ['CG'], 'O1': ['OD1', 'OD2']},
            acceptor_atom_groups={'C2': ['CG'], 'O2': ['OD1', 'OD2']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )

class GluGlu(Side2Side):
    """
    Glutamate's carboxyl group forms a bond with itself.
    """
    _method = "GluGlu"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['GluGlu', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='GLU',
            acceptor_resname='GLU',
            donor_atom_groups={'C1': ['CD'], 'O1': ['OE1', 'OE2']},
            acceptor_atom_groups={'C2': ['CD'], 'O2': ['OE1', 'OE2']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )
    
class Side2Tail(CarboxylicCarbo):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _required_to_tail_keys = ('C2', 'O2') # the keys that must be present in the res_atom_name_dict
    _valid_oxygen_names: Tuple[str, ...] = ('O', 'OXT')

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
                'C1': ['C'],                          # C-terminal carboxyl carbon
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

class AspCTerm(Side2Tail):
    """
    Aspartate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = "AspCTerm"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AspCTerm', ...]:
        return cls._res_to_tail(
            traj=traj,
            res_name='ASP',
            res_atom_name_dict={'C2': ['CG'], 'O2': ['OD1', 'OD2']},
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )
    
class GluCTerm(Side2Tail):
    """
    Glutamate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = "GluCTerm"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['GluCTerm', ...]:
        return cls._res_to_tail(
            traj=traj,
            res_name='GLU',
            res_atom_name_dict={'C2': ['CD'], 'O2': ['OE1', 'OE2']},
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )