from abc import ABCMeta
from typing import final

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers

from ..loss_structure_modifier import LossStructureModifier
from ....core.chemical_loss.chemical_loss import ChemicalLoss, AtomIndexDict
from ....losses.carboxylic_carbo import (
    AspGlu,
    AspAsp,
    GluGlu,
    AspCTerm,
    GluCTerm,
)

class CarboxylicCarboxylicModifier(LossStructureModifier, metaclass=ABCMeta):
    """
    Modify the structure according to the corresponding ChemicalLoss.
    """
    _fragment: str = "c1cc(ccc1C)C" # the fragment to be added to the structure
    _fragment_rdkit_mol: Chem.Mol = Chem.MolFromSmiles(_fragment) # essentially a benzene ring with 2 exterior carbons.

    # Generate 3D coordinates for the fragment at class level
    if _fragment_rdkit_mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        # Add hydrogens temporarily for 3D generation
        # fragment_with_h = Chem.AddHs(_fragment_rdkit_mol)
        # Generate 3D coordinates
        AllChem.EmbedMolecule(_fragment_rdkit_mol, randomSeed=42)
        # Remove hydrogens to get back to original fragment
        #_fragment_rdkit_mol = Chem.RemoveHs(fragment_with_h)

    @property
    def fragment_rdkit_mol(self) -> Chem.Mol:
        return Chem.Mol(self._fragment_rdkit_mol)
    
    @final
    def _inner_mod(self,
                   initial_parsed_mol: Chem.Mol,
                   o1_idx: int, # rdkit atom index
                   o2_idx: int, # rdkit atom index
                   ) -> Chem.Mol:
        """
        To be used in the `_outer_mod` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It also exists soley to organize
        sets of instructions (to make the codebase more readable and modular). It is called `_inner_mod`
        because it is fundamentally what actually modifies the structure. In a way, it is the workhorse
        of all `CarboxylicCarboxylicModifier` subclasses.

        Args:
            initial_parsed_mol: The initial parsed molecule.
            o1_idx: The index of the first oxygen (rdkit atom index) - to be converted to nitrogen.
            o2_idx: The index of the second oxygen (rdkit atom index) - to be converted to nitrogen.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        emol = Chem.EditableMol(initial_parsed_mol)

        # Create nitrogen atoms to replace the oxygens
        n1 = Chem.Atom('N')
        n2 = Chem.Atom('N')
        
        # Replace the oxygen atoms with nitrogen atoms
        emol.ReplaceAtom(o1_idx, n1)
        emol.ReplaceAtom(o2_idx, n2)
        
        # Get the molecule after oxygen replacement
        mol_after_replace = emol.GetMol()
        
        # Combine the modified molecule with the fragment using CombineMols
        combined_mol = Chem.CombineMols(mol_after_replace, self.fragment_rdkit_mol)
        
        # Get the number of atoms in the original molecule after replacement
        num_atoms_original = mol_after_replace.GetNumAtoms()
        
        # Find the outer carbons of the fragment (the two carbons outside the benzene ring)
        # The fragment is "c1cc(ccc1C)C" - benzene ring with two exterior carbons
        outer_carbon_indices = []
        
        # The fragment has 8 atoms: 6 carbons in benzene ring + 2 exterior carbons
        # In the combined molecule, the fragment atoms start at num_atoms_original
        # The exterior carbons are at positions 6 and 7 in the fragment (0-indexed)
        # So in the combined molecule, they're at num_atoms_original + 6 and num_atoms_original + 7
        outer_carbon_1_idx = num_atoms_original + 6  # First exterior carbon
        outer_carbon_2_idx = num_atoms_original + 7  # Second exterior carbon
        
        # Verify these are actually carbons
        if (combined_mol.GetAtomWithIdx(outer_carbon_1_idx).GetAtomicNum() == 6 and 
            combined_mol.GetAtomWithIdx(outer_carbon_2_idx).GetAtomicNum() == 6):
            outer_carbon_indices = [outer_carbon_1_idx, outer_carbon_2_idx]
        else:
            # Fallback: search for carbons that are not in the benzene ring
            benzene_carbons = set()
            for i in range(num_atoms_original, num_atoms_original + 6):  # First 6 atoms are benzene carbons
                benzene_carbons.add(i)
            
            for i in range(num_atoms_original, combined_mol.GetNumAtoms()):
                atom = combined_mol.GetAtomWithIdx(i)
                if (atom.GetAtomicNum() == 6 and i not in benzene_carbons and 
                    len(outer_carbon_indices) < 2):
                    outer_carbon_indices.append(i)
        
        if len(outer_carbon_indices) != 2:
            raise ValueError(f"Expected 2 outer carbons in fragment, found {len(outer_carbon_indices)}")
        
        # Create a new EditableMol from the combined molecule
        emol_combined = Chem.EditableMol(combined_mol)
        
        # Add bonds between nitrogens and outer carbons
        # The nitrogens are at the original o1_idx and o2_idx positions
        emol_combined.AddBond(o1_idx, outer_carbon_indices[0], Chem.BondType.SINGLE)
        emol_combined.AddBond(o2_idx, outer_carbon_indices[1], Chem.BondType.SINGLE)
        
        # Get the final molecule
        final_mol = emol_combined.GetMol()

        # --- Freeze all original atoms and relax only the added fragment and new bonds ---
        # Sanitize the molecule first to fix any valence issues
        Chem.SanitizeMol(final_mol)
        
        # Optimize the molecule with constraints - fix all original atoms
        # Get force field for optimization
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            final_mol, 
            rdForceFieldHelpers.MMFFGetMoleculeProperties(final_mol), 
            confId=0
        )
        
        # Fix all original atoms (atoms from the initial molecule)
        for atom_idx in range(num_atoms_original):
            ff.AddFixedPoint(atom_idx)
        
        # Optimize the molecule
        ff.Minimize(maxIts=200)

        return final_mol
    
    def _debug_zero_vectors(self, mol: Chem.Mol, mol_name: str):
        """
        Debug function to identify atoms with zero-length vectors that cause normalization errors.
        
        Args:
            mol: The molecule to check
            mol_name: Name of the molecule for debugging output
        """
        print(f"\n=== DEBUG: Checking {mol_name} for zero-length vectors ===")
        print(f"Number of atoms: {mol.GetNumAtoms()}")
        print(f"Number of conformers: {mol.GetNumConformers()}")
        
        if mol.GetNumConformers() == 0:
            print("WARNING: No conformers found - this will cause visualization issues!")
            return
        
        conf = mol.GetConformer()
        zero_vector_atoms = []
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            
            # Check if position is at origin (0,0,0) or has very small coordinates
            if (abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6):
                zero_vector_atoms.append((i, atom.GetSymbol(), pos))
                print(f"  Atom {i} ({atom.GetSymbol()}): ZERO VECTOR at {pos}")
            elif (abs(pos.x) < 1e-3 or abs(pos.y) < 1e-3 or abs(pos.z) < 1e-3):
                print(f"  Atom {i} ({atom.GetSymbol()}): SMALL VECTOR at {pos}")
        
        if zero_vector_atoms:
            print(f"  FOUND {len(zero_vector_atoms)} ATOMS WITH ZERO VECTORS!")
            print("  These will cause 'Cannot normalize a zero length vector' errors")
        else:
            print("  No zero-length vectors found")
        
        # Check fragment atoms specifically
        if mol_name == "final_mol":
            num_atoms_original = mol.GetNumAtoms() - self.fragment_rdkit_mol.GetNumAtoms()
            print(f"\n  Fragment atoms (indices {num_atoms_original} to {mol.GetNumAtoms()-1}):")
            for i in range(num_atoms_original, mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                pos = conf.GetAtomPosition(i)
                print(f"    Fragment atom {i} ({atom.GetSymbol()}): {pos}")
        
        print("=" * 60)
    
    @final
    def _outer_mod(self,
                   chemical_loss: ChemicalLoss,
                   initial_parsed_mol: Chem.Mol,
                   mdtraj_atom_indexes_dict: AtomIndexDict,
                   rdkit_atom_indexes_dict: AtomIndexDict,
                   ) -> Chem.Mol:
        """
        To be used in the `_mod_struct` method of the subclasses entirely for utility purposes.
        
        This method is not intended to be overridden by subclasses. It handles the logic for determining
        which oxygen to remove when forming a carboxylic acid, then calls `_inner_mod` to perform the actual
        structural modification.
        
        Args:
            chemical_loss: The chemical loss object containing atom indices.
            initial_parsed_mol: The initial parsed molecule.
            mdtraj_atom_indexes_dict: Dictionary mapping atom names to MDTraj atom indices.
            rdkit_atom_indexes_dict: Dictionary mapping atom names to RDKit atom indices.

        Returns:
            Chem.Mol, the modified RDKit molecule
        """
        inverse_mdtraj_atom_indexes_dict = self._invert_dict(mdtraj_atom_indexes_dict)
        o1_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O1']]
        o2_key = inverse_mdtraj_atom_indexes_dict[chemical_loss._atom_idxs['O2']]

        o1_rdkit_idx = rdkit_atom_indexes_dict[o1_key]
        o2_rdkit_idx = rdkit_atom_indexes_dict[o2_key]
        
        return self._inner_mod(initial_parsed_mol, o1_rdkit_idx, o2_rdkit_idx)
    
    def _mod_struct(self,
                    initial_parsed_mol: Chem.Mol,
                    chemical_loss: ChemicalLoss,
                    mdtraj_atom_indexes_dict: AtomIndexDict,
                    rdkit_atom_indexes_dict: AtomIndexDict,
                    ) -> Chem.Mol:
        """
        """
        return self._outer_mod(
            chemical_loss, 
            initial_parsed_mol, 
            mdtraj_atom_indexes_dict, 
            rdkit_atom_indexes_dict,
            )
    
class CarboxylicCarboSide2SideModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a sidechain carboxyl group.
    """
    pass

class AspGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with Glutamate's carboxyl group.
    """
    _method = AspGlu._method

class AspAspModifier(CarboxylicCarboSide2SideModifier):
    """
    Aspartate's carboxyl group forms a bond with itself.
    """
    _method = AspAsp._method

class GluGluModifier(CarboxylicCarboSide2SideModifier):
    """
    Glutamate's carboxyl group forms a bond with itself.
    """
    _method = GluGlu._method

class CarboxylicCarbo2CTermModifier(CarboxylicCarboxylicModifier):
    """
    Base class for carboxyl-to-carboxyl cyclization chemistries where a sidechain
    carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    pass

class AspCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Aspartate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = AspCTerm._method

class GluCTermModifier(CarboxylicCarbo2CTermModifier):
    """
    Glutamate's carboxyl group forms a bond with a c-terminal carboxyl group.
    """
    _method = GluCTerm._method