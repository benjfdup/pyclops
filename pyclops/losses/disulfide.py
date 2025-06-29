from typing import Optional

import mdtraj as md
import torch

from ..core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ..utils.utils import _inherit_docstring
from .utils.standard_kde_locations import STANDARD_KDE_LOCATIONS


class Disulfide(ChemicalLoss):
    """
    Class for disulfide bond chemistries between two cysteine residues.
    """
    _atom_idxs_keys = ( # DO NOT CHANGE THE ORDER OF THESE KEYS, WILL AFFECT THE KDE CALCULATION & BREAK THE CODE
        'S1',  # AA1 sulfur of the first cysteine
        'C1',  # AA1 carbon bound to S1 (CB)
        'S2',  # AA2 sulfur of the second cysteine
        'C2',  # AA2 carbon bound to S2 (CB)
    )
    _kde_file = STANDARD_KDE_LOCATIONS['disulfide']
    _method = "Disulfide"

    @_inherit_docstring(ChemicalLoss.get_loss_instances)
    @classmethod
    def get_loss_instances(cls,
                           traj: md.Trajectory,
                           atom_indexes_dict: AtomIndexDict,
                           weight: float = 1.0,
                           offset: float = 0.0,
                           temp: float = 1.0,
                           device: Optional[torch.device] = None,
                           ) -> tuple['Disulfide', ...]:
        return cls._get_donor_acceptor_linkages(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_resname='CYS',
            acceptor_resname='CYS',
            donor_atom_groups={'S1': ['SG'], 'C1': ['CB']},
            acceptor_atom_groups={'S2': ['SG'], 'C2': ['CB']},
            weight=weight,
            offset=offset,
            temp=temp,
            device=device,
        )