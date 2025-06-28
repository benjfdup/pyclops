from typing import Optional

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS


class LysTyr(ChemicalLoss):
    """
    Class for lysine-tyrosine chemical interactions.
    """
    _atom_idxs_keys = (
        'N1',  # Nitrogen of the lysine (NZ)
        'C1',  # Delta carbon of the lysine (CD) TODO: double check this is not really CE... but for now I'll trust it.
        'O1',  # Hydroxyl oxygen of the tyrosine (OH)
        'C2',  # Zeta carbon of the tyrosine ring (CZ)
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
                           ) -> tuple['LysTyr', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='LYS',
            acceptor_resname='TYR',
            donor_atom_groups={'N1': ['NZ'], 'C1': ['CD']},
            acceptor_atom_groups={'O1': ['OH'], 'C2': ['CZ']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )