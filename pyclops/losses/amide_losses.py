from typing import Optional, FrozenSet, List, Dict, Sequence
from abc import ABCMeta

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from ..utils.constants import CANONICAL_AMINO_ACID_3_LETTER_CODES, AMBER_CAPS
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS


class Amide(ChemicalLoss, metaclass=ABCMeta):
    """
    Base class for all amide bond cyclization chemistries.

    An amide bond forms between a nitrogen (usually from an amine group)
    and a carbon from a carboxyl group, with the loss of water.
    
    The geometry is defined by four atoms:
    - N1: The nitrogen involved in the amide bond
    - C1: The carbon of the carboxyl group
    - O1: An oxygen from the carboxyl group (potentially resonant)
    - C2: The carbon 'behind' the nitrogen (sometimes the alpha carbon, sometimes a sidechain carbon), 
          in the amide's amino acid
    """
    # required class variables
    _atom_idxs_keys = ('N1', 'C1', 'O1', 'C2')
    _kde_file = STANDARD_KDE_LOCATIONS['amide']

    # helpful class variables unique to amide losses
    _common_caps = AMBER_CAPS
    _valid_oxygen_names = ('O', 'OXT') # perhaps make this a class variable
    _res_to_term_terminal_names = ('head', 'tail') # names of allowed terminal specifications for the `_res_to_term` method
    _required_to_head_keys = (_atom_idxs_keys[1], _atom_idxs_keys[2]) # C1 and O1, because the head is the N-terminal 
                                                                        # (and therefore needs a carboxyl carbon and oxygen to bond with)
    _required_to_tail_keys = (_atom_idxs_keys[0], _atom_idxs_keys[-1]) # N1 and C2, because the tail is the C-terminal 
                                                                        # (and therefore needs a terminal nitrogen and amine carbon to bond with)

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
        elif term_name == 'tail':
            if set(res_atom_name_dict.keys()) != set(cls._required_to_tail_keys):
                raise ValueError(f"""Invalid keys in res_atom_name_dict for 'tail' as term_name: 
                                 {tuple(res_atom_name_dict.keys())}. Must be {cls._required_to_tail_keys}""")
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
                     ) -> tuple['Amide', ...]:
        """
        Helper method to make `get_loss_instances` methods easier to write for losses that connect
        a sidechain to a terminal group via an amide bond. Returns a tuple of `Amide` instances (though
        in reality this will be used by subclasses to return a tuple of `cls` instances).

        Args:
            traj: MDTraj trajectory for analysis and atom identification
            res_name: Name of the residue to connect to the terminal groups
            res_atom_name_dict: Dictionary mapping from _atom_idxs_keys members to possible atom names in the residue 
            that connects to the terminal group. If term_name is 'head', then the keys should be C1 and O1. 
            If term_name is 'tail', then the keys should be N1 and C2.
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
        if len(valid_residues) < 2:
            return ()
        
        losses: List['Amide'] = []
        user_specified_residue_list: List[md.Residue] = [r for r in valid_residues if r.name == res_name]
        if len(user_specified_residue_list) == 0:
            return ()
        
        if term_name == 'head':
            # N-terminal atoms - the head provides N1 and C2
            n_term_residue = valid_residues[0]
            n_term_atom_groups = {
                'N1': ['N'],      # N-terminal nitrogen
                'C2': ['CA'],     # N-terminal alpha carbon
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
                
        elif term_name == 'tail':
            # C-terminal atoms - the tail provides C1 and O1
            c_term_residue = valid_residues[-1]
            c_term_atom_groups = {
                'C1': ['C'],                          # C-terminal carboxyl carbon
                'O1': list(cls._valid_oxygen_names),  # C-terminal carboxyl oxygens
            }
            
            # Find valid C-terminal atoms
            c_term_atoms = cls._find_valid_atoms_for_residue(
                c_term_residue.index, c_term_atom_groups, atom_indexes_dict
            )
            if not c_term_atoms:
                return ()
            
            # Process each user-specified residue for N1 and C2
            for user_specified_residue in user_specified_residue_list:
                # Find valid atoms for the user-specified residue
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
        else:
            raise ValueError(f"Invalid term_name: {term_name}. Must be one of {cls._res_to_term_terminal_names}")
        
        return tuple(losses)

# head2tail
class AmideHead2Tail(Amide):
    """
    Backbone-to-backbone amide bond between the N-terminus and C-terminus.
    
    This represents a standard head-to-tail cyclization where the C-terminal
    carboxyl group forms an amide bond with the N-terminal amine.
    """
    _method = "AmideHead2Tail"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideHead2Tail', ...]:
        
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
        
        valid_residues = [r for r in all_residues if r.name not in exclude_residue_names]
        
        # Need at least 2 residues for head-to-tail connection
        if len(valid_residues) < 2:
            return ()
        
        # Get first (N-terminal) and last (C-terminal) residues
        n_term_residue = valid_residues[0]
        c_term_residue = valid_residues[-1]
        
        # Find required atoms
        # N-terminal atoms (acceptor)
        n1_idx = atom_indexes_dict.get((n_term_residue.index, 'N'))  # N-terminal nitrogen
        c2_idx = atom_indexes_dict.get((n_term_residue.index, 'CA'))  # Alpha carbon
        
        # C-terminal atoms (donor)
        c1_idx = atom_indexes_dict.get((c_term_residue.index, 'C'))  # Carboxyl carbon
        
        # Check if core atoms exist
        if any(idx is None for idx in [n1_idx, c1_idx, c2_idx]):
            return ()
        
        # Find all available resonant oxygen atoms
        oxygen_idxs = []
        for oxygen_name in cls._valid_oxygen_names:
            oxygen_idx = atom_indexes_dict.get((c_term_residue.index, oxygen_name))
            if oxygen_idx is not None:
                oxygen_idxs.append(oxygen_idx)
        
        # If no oxygen atoms found, return empty
        if not oxygen_idxs:
            return ()
        
        # Create separate loss instances for each resonant form
        result = []
        resonance_key_base: FrozenSet[int] = frozenset([n_term_residue.index, c_term_residue.index])
        
        for oxygen_idx in oxygen_idxs:
            # Create atom indices dictionary for this resonant form
            atom_idxs = {
                'N1': n1_idx,      # N-terminal nitrogen
                'C1': c1_idx,      # C-terminal carboxyl carbon
                'O1': oxygen_idx,  # C-terminal carboxyl oxygen (potentially resonant)
                'C2': c2_idx       # N-terminal alpha carbon
            }
            
            # Create resonance key if there are multiple oxygen atoms
            resonance_key = (cls._method, resonance_key_base)
            
            # Create the ChemicalLoss instance
            instance = cls(
                atom_idxs=atom_idxs,
                temp=temp,
                weight=weight,
                offset=offset,
                resonance_key=resonance_key,
                device=device
            )
            
            result.append(instance)
        
        return tuple(result)
    
# side2side
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
    _method = "AmideLysGlu"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideLysGlu', ...]:
        
        # Define donor (lysine) and acceptor (glutamate) atom groups
        donor_atom_groups = {
            'N1': ['NZ'],        # Lysine's terminal nitrogen
            'C2': ['CE'],        # Lysine's carbon behind the nitrogen
        }
        
        acceptor_atom_groups = {
            'C1': ['CD'],        # Glutamate's carboxyl carbon
            'O1': ['OE1', 'OE2'], # Glutamate's carboxyl oxygens (resonant forms)
        }
        
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='GLU', 
            donor_atom_groups=donor_atom_groups,
            acceptor_atom_groups=acceptor_atom_groups,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
class AmideLysAsp(AmideSide2Side):
    """
    Amide bond between Lysine's sidechain NH3+ and Aspartate's sidechain COO-.
    
    This represents a specific sidechain-to-sidechain cyclization where
    lysine's terminal amine forms an amide bond with aspartate's carboxyl group.
    """
    _method = "AmideLysAsp"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideLysAsp', ...]:
        
        # Define donor (lysine) and acceptor (glutamate) atom groups
        donor_atom_groups = {
            'N1': ['NZ'],        # Lysine's terminal nitrogen
            'C2': ['CE'],        # Lysine's carbon behind the nitrogen
        }

        acceptor_atom_groups = {
            'C1': ['CG'],              # Aspartate carboxyl carbon
            'O1': ['OD1', 'OD2'],      # Carboxyl oxygens (resonant)
        }

        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='ASP', 
            donor_atom_groups=donor_atom_groups,
            acceptor_atom_groups=acceptor_atom_groups,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

class AmideOrnGlu(AmideSide2Side):
    """
    Amide bond between Ornithine's sidechain NH3+ and Glutamate's sidechain COO-.
    
    This represents a specific sidechain-to-sidechain cyclization where
    ornithine's terminal amine forms an amide bond with glutamate's carboxyl group.
    """
    _method = "AmideOrnGlu"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideOrnGlu', ...]:
        
        # Define donor (ornithine) and acceptor (glutamate) atom groups
        donor_atom_groups = {
            'N1': ['NE'],        # Ornithine's terminal nitrogen
            'C2': ['CD'],        # Ornithine's carbon behind the nitrogen
        }
        
        acceptor_atom_groups = {
            'C1': ['CD'],        # Glutamate's carboxyl carbon
            'O1': ['OE1', 'OE2'], # Glutamate's carboxyl oxygens (resonant forms)
        }
        
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='ORN',
            acceptor_resname='GLU', 
            donor_atom_groups=donor_atom_groups,
            acceptor_atom_groups=acceptor_atom_groups,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
# side2head
class AmideSide2Head(Amide):
    """
    Base class for sidechain-to-N-terminal amide bond cyclization.
    
    This represents amide bonds formed between a sidechain amine group
    and the N-terminal amine group.
    """
    pass

class AmideAspHead(AmideSide2Head):
    """
    Amide bond between Aspartate's sidechain COO- and the N-terminal amine group.
    
    This represents a specific sidechain-to-N-terminal amide bond cyclization where
    aspartate's carboxyl group forms an amide bond with the N-terminal amine group.
    """
    _method = "AmideAspHead"
    
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideAspHead', ...]:
        
        return cls._res_to_term(
            traj=traj,
            res_name='ASP',
            res_atom_name_dict={'C1': ['CG'], 'O1': ['OD1', 'OD2']},
            term_name='head',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )
    
class AmideGluHead(AmideSide2Head):
    """
    Amide bond between Glutamate's sidechain COO- and the N-terminal amine group.
    
    This represents a specific sidechain-to-N-terminal amide bond cyclization where
    glutamate's carboxyl group forms an amide bond with the N-terminal amine group.
    """
    _method = "AmideGluHead"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideGluHead', ...]:
        
        return cls._res_to_term(
            traj=traj,
            res_name='GLU',
            res_atom_name_dict={'C1': ['CD'], 'O1': ['OE1', 'OE2']},
            term_name='head',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )

# side2tail
class AmideSide2Tail(Amide):
    """
    Base class for sidechain-to-N-terminal amide bond cyclization.
    
    This represents amide bonds formed between a sidechain amine group
    and the N-terminal amine group.
    """
    pass

class AmideLysTail(AmideSide2Tail):
    """
    Amide bond between Lysine's sidechain NH3+ and the C-terminal carboxyl group.
    
    This represents a specific sidechain-to-N-terminal amide bond cyclization where
    lysine's terminal amine forms an amide bond with the C-terminal carboxyl group.
    """
    _method = "AmideLysTail"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideLysTail', ...]:
        
        return cls._res_to_term(
            traj=traj,
            res_name='LYS',
            res_atom_name_dict={'N1': ['NZ'], 'C2': ['CE']},
            term_name='tail',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )
    
class AmideArgTail(AmideSide2Tail):
    """
    Amide bond between Arginine's sidechain NH3+ and the C-terminal carboxyl group.
    
    This represents a specific sidechain-to-N-terminal amide bond cyclization where
    arginine's terminal amine forms an amide bond with the C-terminal carboxyl group.
    """
    _method = "AmideArgTail"
    
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideArgTail', ...]:
        return cls._res_to_term(
            traj=traj,
            res_name='ARG',
            res_atom_name_dict={'N1': ['NH1', 'NH2'], 'C2': ['CZ']},
            term_name='tail',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )
    
class AmideOrnTail(AmideSide2Tail):
    """
    Amide bond between Ornithine's sidechain NH3+ and the C-terminal carboxyl group.
    
    This represents a specific sidechain-to-N-terminal amide bond cyclization where
    ornithine's terminal amine forms an amide bond with the C-terminal carboxyl group.
    """
    _method = "AmideOrnTail"
    
    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls, 
                           traj: md.Trajectory, 
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['AmideOrnTail', ...]:
        return cls._res_to_term(
            traj=traj,
            res_name='ORN',
            res_atom_name_dict={'N1': ['NE'], 'C2': ['CD']},
            term_name='tail',
            atom_indexes_dict=atom_indexes_dict,
            weight=weight,
            offset=offset,
            temp=temp,
            device=device
        )