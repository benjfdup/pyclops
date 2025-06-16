from typing import Dict, List

import mdtraj as md
import parmed as pmd

from ..core.chemical_loss import ChemicalLoss
from ..utils.indexing import IndexesMethodPair
from .standard_file_locations import STANDARD_KDE_LOCATIONS, STANDARD_LINKAGE_PDB_LOCATIONS

class Disulfide(ChemicalLoss):
    """
    Models disulfide bond formation between two cysteine residues.
    
    This loss function guides the geometry of two cysteine side chains to form
    configurations favorable for disulfide bond formation. Disulfide bonds are
    crucial for protein tertiary structure stabilization and can significantly
    impact protein folding and stability.
    
    The tetrahedral geometry is defined by:
    - The sulfur atoms of both cysteines (SG)
    - The beta carbons of both cysteines (CB)
    
    This evaluates the proper distance and angle constraints for a chemically
    realistic disulfide bond geometry.
    
    Attributes:
        atom_idxs_keys (List[str]): The four atoms defining the tetrahedral geometry:
            - 'S1': Sulfur atom of the first cysteine (SG)
            - 'S2': Sulfur atom of the second cysteine (SG)
            - 'C1': Beta carbon of the first cysteine (CB)
            - 'C2': Beta carbon of the second cysteine (CB)
        kde_file (str): Path to the KDE model file (.pt) for the statistical potential
    """
    
    atom_idxs_keys = [
        'S1',  # sulfur of the first cysteine
        'S2',  # sulfur of the second cysteine
        'C1',  # carbon bound to S1 (CB)
        'C2',  # carbon bound to S2 (CB)
    ]

    kde_file = STANDARD_KDE_LOCATIONS['disulfide']
    linkage_pdb_file = STANDARD_LINKAGE_PDB_LOCATIONS['disulfide']


    @classmethod
    def get_indexes_and_methods(cls, 
                                traj: md.Trajectory, 
                                atom_indexes_dict: Dict) -> List[IndexesMethodPair]:
        """
        Find all valid cysteine-cysteine pairings for disulfide bond formation.
        
        This method identifies all possible pairs of cysteines in the structure that
        could potentially form disulfide bonds, and creates the appropriate atom
        selections for evaluating the disulfide geometry.
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory containing residue and atom information.
        
        atom_indexes_dict : Dict[Tuple[int, str], int]
            Dictionary mapping (residue_index, atom_name) to atom index.
            
        Returns
        -------
        List[IndexesMethodPair]
            List of valid index-method pairs for all potential disulfide bonds.
            Each pair contains:
            - Dictionary mapping from atom keys to atom indices
            - Method string describing the specific disulfide bond
            - Set of involved residue indices
            
        Notes
        -----
        Only considers unique pairs of cysteines (no self-pairing or duplicate pairs).
        """
        
        def disulfide_pair_selection(donor_residues, acceptor_residues):
            """Create unique pairs of cysteines, avoiding duplicates and self-pairing."""
            # Since donor_residues and acceptor_residues are the same (all CYS residues),
            # we need to create unique pairs without duplicates
            pairs = []
            for i, cys1 in enumerate(donor_residues):
                for cys2 in donor_residues[i+1:]:  # Only consider each pair once
                    pairs.append((cys1, cys2))
            return pairs
        
        return cls.find_valid_pairs(
            traj=traj,
            atom_indexes_dict=atom_indexes_dict,
            donor_residue_names="CYS",
            acceptor_residue_names="CYS",
            donor_atom_groups={
                'S1': ['SG'],
                'C1': ['CB']
            },
            acceptor_atom_groups={
                'S2': ['SG'],
                'C2': ['CB']
            },
            method_name="Disulfide",
            special_selection=disulfide_pair_selection
        )
    
    def _build_final_structure(self, 
                              initial_structure: pmd.Structure) -> pmd.Structure:
        """
        Builds a final structure with the disulfide bond formed.
        
        This method:
        1. Identifies the sulfur atoms involved in the disulfide bond
        2. Finds and removes hydrogen atoms attached to those sulfurs
        3. Creates a disulfide bond between the sulfur atoms
        4. Updates the topology to reflect the new connectivity
        
        Args:
            initial_structure: ParmED Structure object representing the initial state
            
        Returns:
            ParmED Structure object with the disulfide bond formed
        """
        # STEPS:
        # 1. Remove hydrogens from the sulfur atoms of the relevant cysteines 
        # of the initial structure (we can get this info from self.atom_idxs)
        # 2. Form a disulfide bond between the sulfur atoms
        # 3. Return the final structure
        
        # Create a copy to avoid modifying the original
        final_structure = initial_structure.copy()
        
        # Get the sulfur atom indices from our atom mapping (these are the SPECIFIC sulfurs for this bond)
        s1_idx = self._atom_idxs['S1']
        s2_idx = self._atom_idxs['S2']
        
        # Get the actual atoms and keep references to them
        s1_atom = final_structure.atoms[s1_idx]
        s2_atom = final_structure.atoms[s2_idx]
        
        # Verify these are actually sulfur atoms in cysteine residues
        if s1_atom.element_symbol != 'S' or s2_atom.element_symbol != 'S':
            raise ValueError(f"Expected sulfur atoms, got {s1_atom.element_symbol} and {s2_atom.element_symbol}")
        
        if s1_atom.residue.name != 'CYS' or s2_atom.residue.name != 'CYS':
            raise ValueError(f"Expected cysteine residues, got {s1_atom.residue.name} and {s2_atom.residue.name}")
        
        # Remove hydrogen atoms bonded to these specific sulfur atoms (but don't remake yet)
        final_structure = self._remove_hydrogens_from_atoms(final_structure, 
                                                            [s1_idx, s2_idx], 
                                                            remake=False)
        
        # Create the disulfide bond using the original atom references (still valid since no remake)
        disulfide_bond = pmd.Bond(s1_atom, s2_atom)
        final_structure.bonds.append(disulfide_bond)
        
        # Update residue names to reflect disulfide bonding (CYX for disulfide-bonded cysteine)
        s1_atom.residue.name = 'CYX'
        s2_atom.residue.name = 'CYX'
        
        # Now remake the structure to finalize all changes
        final_structure.remake()
        
        return final_structure