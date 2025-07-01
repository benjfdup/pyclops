from typing import Optional

from rdkit import Chem

from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.lys_arg import LysArg
from ..loss_structure_modifier import LossStructureModifier

class LysArgModifier(LossStructureModifier):
    """
    Class for lysine-arginine chemical interactions.
    """
    _method = LysArg._method

    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   n1_idx: int, # rdkit atom index, nitrogen of the lysine

                   n2_idx: int, # rdkit atom index, one outer nitrogen of the arginine (this one has the double bond)
                   n3_idx: int, # rdkit atom index, other outer nitrogen of the arginine
                   c1_idx: int, # rdkit atom index, central carbon of the arginine (CZ)
                   n4_idx: int, # rdkit atom index, inner nitrogen of the arginine
                   ) -> Chem.Mol:
        """
        Create a lysine-arginine chemical interaction by modifying the molecular structure.
        
        Args:
            initial_parsed_mol: The initial parsed molecule
            n1_idx: RDKit atom index of the lysine nitrogen (NZ)
            n2_idx: RDKit atom index of one outer nitrogen of arginine (NH1/NH2)
            n3_idx: RDKit atom index of the other outer nitrogen of arginine (NH2/NH1)
            c1_idx: RDKit atom index of the central carbon of arginine (CZ)
            n4_idx: RDKit atom index of the inner nitrogen of arginine (NE)
            
        Returns:
            Modified RDKit molecule with lysine-arginine interaction
        """
        # Create editable molecule
        emol = Chem.EditableMol(initial_parsed_mol)
        
        # Step 1: Add 2 new carbon atoms bonded to n1 (lysine nitrogen)
        # Add first carbon atom
        c2_idx = emol.AddAtom(Chem.Atom(6))  # Atomic number 6 is carbon
        
        # Add second carbon atom  
        c3_idx = emol.AddAtom(Chem.Atom(6))  # Atomic number 6 is carbon
        
        # Bond the new carbons to n1
        emol.AddBond(n1_idx, c2_idx, order=Chem.rdchem.BondType.SINGLE)
        emol.AddBond(n1_idx, c3_idx, order=Chem.rdchem.BondType.SINGLE)
        
        # Step 2: Bond the first new carbon to n2, the second to n3
        emol.AddBond(c2_idx, n2_idx, order=Chem.rdchem.BondType.SINGLE)
        emol.AddBond(c3_idx, n3_idx, order=Chem.rdchem.BondType.SINGLE)
        
        # Step 3: Change the bond between n4 and c1 to a single bond
        # First, remove the existing bond
        bond = initial_parsed_mol.GetBondBetweenAtoms(n4_idx, c1_idx)
        if bond is not None:
            emol.RemoveBond(n4_idx, c1_idx)
            # Add it back as a single bond
            emol.AddBond(n4_idx, c1_idx, order=Chem.rdchem.BondType.SINGLE)
        
        # Step 4: Change the bond between n2 and c1 to a double bond
        # First, remove the existing bond
        bond = initial_parsed_mol.GetBondBetweenAtoms(n2_idx, c1_idx)
        if bond is not None:
            emol.RemoveBond(n2_idx, c1_idx)
            # Add it back as a double bond
            emol.AddBond(n2_idx, c1_idx, order=Chem.rdchem.BondType.DOUBLE)
        
        # Get the modified molecule and sanitize it
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        
        return new_mol
    
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ) -> Chem.Mol:
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which nitrogen to use for the double bond, then calls `_inner_mod` to perform the actual
        structural modification.
        """
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)

        method_str, idxs_set = chemical_loss.resonance_key

        arg_idx: Optional[int] = None
        lys_idx: Optional[int] = None
        for aa_i in idxs_set:
            ca_key = (aa_i, 'CA')
            ca_idx = rdkit_atom_indexes_dict[ca_key]
            if initial_parsed_mol.GetAtomWithIdx(ca_idx).GetPDBResidueInfo().GetResidueName() == 'ARG':
                arg_idx = aa_i
            elif initial_parsed_mol.GetAtomWithIdx(ca_idx).GetPDBResidueInfo().GetResidueName() == 'LYS':
                lys_idx = aa_i
        if arg_idx is None or lys_idx is None:
            raise ValueError(f"Could not find arginine or lysine in the molecule for the chemical loss {chemical_loss.resonance_key}")
        
        n1_idx = rdkit_atom_indexes_dict[(lys_idx, 'NZ')]

        n2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['N2']]
        n2_idx = rdkit_atom_indexes_dict[n2_key]

        n3_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['N3']]
        n3_idx = rdkit_atom_indexes_dict[n3_key]

        n4_idx = rdkit_atom_indexes_dict[(arg_idx, 'NE')]
        c1_idx = rdkit_atom_indexes_dict[(arg_idx, 'CZ')]
        
        return self._inner_mod(initial_parsed_mol,
                               n1_idx,
                               n2_idx,
                               n3_idx,
                               c1_idx,
                               n4_idx)
    
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        """
        return self._outer_mod(chemical_loss, 
                               initial_parsed_mol,
                               mdtraj_atom_indexes_dict, 
                               rdkit_atom_indexes_dict)