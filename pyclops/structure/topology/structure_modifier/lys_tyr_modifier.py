from typing import final

from rdkit import Chem

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.lys_tyr import LysTyr
from ..loss_structure_modifier import LossStructureModifier

@final
class LysTyrModifier(LossStructureModifier):
    """
    Class for lysine-tyrosine chemical interactions.
    """
    _method = LysTyr._method

    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   n1_idx: int, # Nitrogen of the lysine (NZ)
                   c2_idx: int, # CE1 or CE2 of the tyrosine ring (this is potentially resonant)
                   ) -> Chem.Mol:
        """
        Create a lysine-tyrosine chemical interaction by adding a bridging carbon atom.
        
        Args:
            initial_parsed_mol: The initial parsed molecule.
            n1_idx: The index of the nitrogen atom of the lysine (NZ).
            c2_idx: The index of the carbon atom of the tyrosine ring (CE1 or CE2).
            
        Returns:
            Modified RDKit molecule with the lysine-tyrosine interaction.
        """
        emol = Chem.EditableMol(initial_parsed_mol)
        
        # Step 1: Add a new carbon atom bound to n1
        new_carbon_idx = emol.AddAtom(Chem.Atom(6))  # Atomic number 6 is carbon
        emol.AddBond(n1_idx, new_carbon_idx, order=Chem.rdchem.BondType.SINGLE)
        
        # Step 2: Bind that new carbon to c2
        emol.AddBond(new_carbon_idx, c2_idx, order=Chem.rdchem.BondType.SINGLE)
        
        new_mol = emol.GetMol()
        
        Chem.SanitizeMol(new_mol)
        
        # Step 3: Relax the newly added carbon position
        final_mol = self._relax_atom_subset(new_mol, [new_carbon_idx])
        
        return final_mol
    
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ) -> Chem.Mol:
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which carbon to use for the bridging carbon, then calls `_inner_mod` to perform the actual
        structural modification.
        """
        # Convert MDTraj atom indices to RDKit atom indices
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        
        n1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['N1']]
        c2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['C2']]

        n1_rdkit_idx = rdkit_atom_indexes_dict[n1_key]
        c2_rdkit_idx = rdkit_atom_indexes_dict[c2_key]

        return self._inner_mod(initial_parsed_mol, n1_rdkit_idx, c2_rdkit_idx)

    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        Modify the structure according to the corresponding ChemicalLoss.
        """
        return self._outer_mod(chemical_loss, 
                               initial_parsed_mol,
                               mdtraj_atom_indexes_dict, 
                               rdkit_atom_indexes_dict)