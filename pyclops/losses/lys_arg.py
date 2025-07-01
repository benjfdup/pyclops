from typing import Optional, Tuple, final

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS

@final
class LysArg(ChemicalLoss):
    """
    Class for lysine-arginine chemical interactions.
    """
    _atom_idxs_keys = ( # DO NOT CHANGE THE ORDER OF THESE KEYS, WILL AFFECT THE KDE CALCULATION & BREAK THE CODE
        'N1',  # AA1 Nitrogen of the Lysine (NZ)
        'N2',  # AA2 One outer nitrogen of the arginine (NH1/NH2, resonant with N3) (this one gets the double bond)
        'N3',  # AA2 The other outer nitrogen of the arginine (NH2/NH1, resonant with N2)
        'N4',  # AA2 The "inner" nitrogen of the arginine (NE)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['lys-arg']
    _method = "LysArg"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> Tuple['LysArg', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='ARG',
            donor_atom_groups={'N1': ['NZ']},
            acceptor_atom_groups={'N2': ['NH1', 'NH2'], 'N3': ['NH2', 'NH1'], 'N4': ['NE']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )