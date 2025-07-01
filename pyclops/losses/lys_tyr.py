from typing import Optional, Tuple, final

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS

@final
class LysTyr(ChemicalLoss):
    """
    Class for lysine-tyrosine chemical interactions.
    """
    _atom_idxs_keys = ( # DO NOT CHANGE THE ORDER OF THESE KEYS, WILL AFFECT THE KDE CALCULATION & BREAK THE CODE
        'N1',  # AA1 Nitrogen of the lysine (NZ)
        'C1',  # AA1 CE of the lysine (behind the nitrogen)

        'O1',  # AA2 Hydroxyl oxygen of the tyrosine (OH)
        'C2',  # AA2 CE1 or CE2 of the tyrosine ring (this is potentially resonant)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['lys-tyr']
    _method = "LysTyr"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> Tuple['LysTyr', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='TYR',
            donor_atom_groups={'N1': ['NZ'], 'C1': ['CD']},
            acceptor_atom_groups={'O1': ['OH'], 'C2': ['CE1', 'CE2', 'CEx', 'CEy', 'CE%']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )