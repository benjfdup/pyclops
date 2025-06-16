# fundamentally, we want a class which will take in a chemical loss and a structure, of some kind, 
# and return a version of that structure with the chemical loss applied, which we can then minimize.

# this class would probably have a way to index what it needs to do for any given chemical loss, and
# then applies the appropriate steps to the structure, if that makes sense.
import parmed as pmd

from ..core.chemical_loss import ChemicalLoss
from ..losses.standard_file_locations import STANDARD_LINKAGE_PDB_LOCATIONS
from ..utils.constants import AMBER_CAPS

class StructureMaker():
    @staticmethod
    def _remove_amber_caps(initial_structure: pmd.Structure,
                           remove_cap_str: str, 
                           remake: bool = True, 
                           verbose: bool = False,
                           ) -> pmd.Structure:
        """
        Removes Amber cap atoms (ACE, NME, NHE) from the structure based on the specified location.
        
        Args:
            initial_structure: ParmED Structure object to modify
            remove_cap_str: String indicating which caps to remove ('head', 'tail', or 'both')
            remake: Whether to rebuild the structure after modifications
            verbose: Whether to print debug information
            
        Returns:
            Modified ParmED Structure object
            
        Raises:
            ValueError: If remove_cap_str is not one of the valid options
        """
        valid_r_c_s_vals = ['head', 'tail', 'both']
        if remove_cap_str not in valid_r_c_s_vals:
            raise ValueError(f'remove_cap_str must be in {valid_r_c_s_vals}. Got {remove_cap_str}')
            
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
            
        # Track atoms to remove
        atoms_to_remove = []
        
        # Get first and last residues
        first_residue = structure.residues[0]
        last_residue = structure.residues[-1]
        
        # Handle head cap removal
        if remove_cap_str in ['head', 'both']:
            if first_residue.name in AMBER_CAPS:
                atoms_to_remove.extend(list(first_residue.atoms))
                if verbose:
                    print(f"Removing head cap: {first_residue.name}")
        
        # Handle tail cap removal
        if remove_cap_str in ['tail', 'both']:
            if last_residue != first_residue and last_residue.name in AMBER_CAPS:
                atoms_to_remove.extend(list(last_residue.atoms))
                if verbose:
                    print(f"Removing tail cap: {last_residue.name}")
        
        # Remove atoms in reverse order to maintain indices
        for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
            structure.atoms.pop(atom.idx)
        
        # Rebuild the structure to update indices after atom removal
        if remake:
            structure.remake()
        
        return structure
    
    def make_topology(init_struct: pmd.Structure, chem_loss: ChemicalLoss):
        """
        Given an input structure, and the corresponding chemical loss, outputs a new structure with
        bond topology that corresponds to that chemical loss being applied, as specified in its linkage pdb.

        Does not yet regard particular coordinates.
        """
        pass

'''
    @abstractmethod
    def _build_final_structure(self,
                              initial_structure: pmd.Structure) -> pmd.Structure:
        """
        Builds a final structure, assuming this cyclization has already occurred. 
        
        This method should:
        1. Create the cyclization bond(s) using the linkage information
        2. Remove/modify atoms as needed for the specific chemistry
        3. Update the topology to reflect the new connectivity
        4. Preserve coordinates where possible
        
        The returned structure will likely need geometry optimization but should have
        correct connectivity and topology.
        
        Args:
            initial_structure: ParmED Structure object representing the initial state
            
        Returns:
            ParmED Structure object with the cyclization applied
            
        Note:
            This structure will almost certainly need to be optimized, and this will not be handled here.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement _build_final_structure")

    @final
    def _validate_initial_structure(self, initial_structure: pmd.Structure) -> None:
        """
        Validates the initial ParmED structure.
        
        Args:
            initial_structure: ParmED Structure to validate
            
        Raises:
            ValueError: If the structure is invalid for cyclization
            TypeError: If the input is not a ParmED Structure
        """
        if not isinstance(initial_structure, pmd.Structure):
            raise TypeError("Initial structure must be a ParmED Structure object")
        
        if len(initial_structure.atoms) == 0:
            raise ValueError("Initial structure contains no atoms")
            
        # Check that required atoms exist
        atom_indices = [self._atom_idxs[key] for key in self.atom_idxs_keys]
        max_idx = max(atom_indices)
        if max_idx >= len(initial_structure.atoms):
            raise ValueError(f"Atom index {max_idx} exceeds structure size ({len(initial_structure.atoms)} atoms)")
        
    @staticmethod
    def _remove_hydrogens_from_atoms(initial_structure: pmd.Structure, 
                                     atom_idxs: List[int],
                                     remake: bool = True) -> pmd.Structure:
        """
        Removes hydrogens from the relevant atoms of the structure.

        Then remake the structure to update indices after atom removal.
        """
        # Collect hydrogen atoms to remove and their associated bonds
        hydrogens_to_remove = []
        bonds_to_remove = []
        
        # 1. Find all hydrogen atoms bonded to the input atoms
        for atom_idx in atom_idxs:
            for bonded_atom in initial_structure.atoms[atom_idx].bond_partners:
                if bonded_atom.element_symbol == 'H':
                    hydrogens_to_remove.append(bonded_atom)
                    
                    # Find all bonds involving this hydrogen atom
                    for bond in initial_structure.bonds:
                        if bonded_atom in (bond.atom1, bond.atom2):
                            bonds_to_remove.append(bond)
        
        # 2. Remove bonds first (to avoid reference issues)
        for bond in bonds_to_remove:
            if bond in initial_structure.bonds:
                initial_structure.bonds.remove(bond)
        
        # 3. Remove hydrogen atoms (in reverse order to maintain indices)
        for h_atom in sorted(hydrogens_to_remove, key=lambda x: x.idx, reverse=True):
            initial_structure.atoms.pop(h_atom.idx)
        
        # 4. Rebuild the structure to update indices after atom removal
        if remake:
            initial_structure.remake()

        return initial_structure
    
    @final 
    def build_final_structure(self,
                              initial_structure: pmd.Structure) -> pmd.Structure:
        """
        Builds a final structure, assuming this cyclization has already occurred.
        
        This is the public interface that adds validation and error handling around
        the abstract _build_final_structure method.
        
        Args:
            initial_structure: ParmED Structure object representing the initial state
            
        Returns:
            ParmED Structure object with the cyclization applied and validated
            
        Raises:
            ValueError: If the initial structure is invalid
            TypeError: If the input is not a ParmED Structure
        """
        self._validate_initial_structure(initial_structure)
        
        try:
            final_structure = self._build_final_structure(initial_structure)
            
            # Validate the output
            if not isinstance(final_structure, pmd.Structure):
                raise TypeError("_build_final_structure must return a ParmED Structure object")
                
            return final_structure
            
        except Exception as e:
            raise RuntimeError(f"Failed to build final structure for {self.__class__.__name__}: {str(e)}") from e
    
    @staticmethod
    def structure_from_trajectory(traj: md.Trajectory, frame: int = 0) -> pmd.Structure:
        """
        Utility method to convert an MDTraj trajectory to a ParmED Structure.
        
        Args:
            traj: MDTraj trajectory
            frame: Frame index to extract (default: 0)
            
        Returns:
            ParmED Structure object
        """
        if frame >= traj.n_frames:
            raise ValueError(f"Frame {frame} exceeds trajectory length ({traj.n_frames} frames)")
            
        # Use temporary directory for safe file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdb = os.path.join(temp_dir, "temp_structure.pdb")
            traj[frame].save_pdb(temp_pdb)
            structure = pmd.load_file(temp_pdb)
            return structure
    
    @staticmethod  
    def trajectory_from_structure(structure: pmd.Structure) -> md.Trajectory:
        """
        Utility method to convert a ParmED Structure to an MDTraj trajectory.
        
        Args:
            structure: ParmED Structure object
            
        Returns:
            MDTraj trajectory (single frame)
        """
        # Use temporary directory for safe file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdb = os.path.join(temp_dir, "temp_trajectory.pdb")
            structure.save(temp_pdb)
            traj = md.load(temp_pdb)
            return traj
        
    @staticmethod
    def structure_to_pdb(structure: pmd.Structure, filename: str) -> None:
        """
        Utility method to convert a ParmED Structure to a PDB file.
        """
        structure.save(filename, format='pdb')
    
    # these are sloppy as we reuse alot of code. Let's clean this up later.
    @staticmethod
    def _remove_amber_caps(initial_structure: pmd.Structure, 
                           remake: bool = True,
                           verbose: bool = False) -> pmd.Structure:
        """
        Removes Amber cap atoms if they are present (ACE, NME, NHE)
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get first and last residues
        first_residue = structure.residues[0]
        last_residue = structure.residues[-1]
        
        # Track atoms to remove
        atoms_to_remove = []
        
        # Check if first residue is an amber cap
        if first_residue.name in AMBER_CAPS:
            atoms_to_remove.extend(list(first_residue.atoms))
        
        # Check if last residue is an amber cap (and not the same as first)
        if last_residue != first_residue and last_residue.name in AMBER_CAPS:
            atoms_to_remove.extend(list(last_residue.atoms))
        
        # Remove atoms in reverse order to maintain indices
        for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
            structure.atoms.pop(atom.idx)
        
        # Rebuild the structure to update indices after atom removal
        if remake:
            structure.remake()
        
        return structure
    
    @staticmethod
    def _remove_amber_head(initial_structure: pmd.Structure, 
                           remake: bool = True, 
                           verbose: bool = False,
                           ) -> pmd.Structure:
        """
        Removes any Amber caps found in the first residue
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get the first residue
        first_residue = structure.residues[0]
        
        # Check if first residue is an amber cap
        if first_residue.name in AMBER_CAPS:
            # Get all atoms in the first residue
            atoms_to_remove = list(first_residue.atoms)
            
            # Remove atoms in reverse order to maintain indices
            for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
                structure.atoms.pop(atom.idx)
            
            # Rebuild the structure to update indices after atom removal
            if remake:
                structure.remake()
        
        return structure
    
    # disulfide
    @staticmethod
    def _remove_amber_tail(initial_structure: pmd.Structure, 
                           remake: bool = True, 
                           verbose: bool = False,
                           ) -> pmd.Structure:
        """
        Removes any Amber caps found in the last residue
        """
        # Create a copy to avoid modifying the original
        structure = initial_structure.copy()
        
        # If no residues, return the structure as-is
        if not structure.residues:
            if verbose:
                print("No residues found in the structure")
            return structure
        
        # Get the last residue
        last_residue = structure.residues[-1]
        
        # Check if last residue is an amber cap
        if last_residue.name in AMBER_CAPS:
            # Get all atoms in the last residue
            atoms_to_remove = list(last_residue.atoms)
            
            # Remove atoms in reverse order to maintain indices
            for atom in sorted(atoms_to_remove, key=lambda x: x.idx, reverse=True):
                structure.atoms.pop(atom.idx)
            
            # Rebuild the structure to update indices after atom removal
            if remake:
                structure.remake()
        
        return structure

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

# amide
def _bfs(self, 
             initial_structure: pmd.Structure, 
             remove_amber_str: Optional[str], # "head", "tail", "caps" or None
             ) -> pmd.Structure:
        # Create a copy to avoid modifying the original
        final_structure = initial_structure.copy()
        
        # Get the nitrogen and carbon atom indices from our atom mapping
        n1_idx = self._atom_idxs['N1']  # Nitrogen involved in the amide bond
        c1_idx = self._atom_idxs['C1']  # Carbon of the carboxyl group
        o1_idx = self._atom_idxs['O1']  # Oxygen of the carboxyl group
        c2_idx = self._atom_idxs['C2']  # Carbon 'behind' the nitrogen
        
        # Get the actual atoms and keep references to them
        n1_atom = final_structure.atoms[n1_idx]
        c1_atom = final_structure.atoms[c1_idx]
        o1_atom = final_structure.atoms[o1_idx]
        c2_atom = final_structure.atoms[c2_idx]
        
        # Verify these are the expected atom types
        if n1_atom.element_symbol != 'N':
            raise ValueError(f"Expected nitrogen atom for N1, got {n1_atom.element_symbol}")
        if c1_atom.element_symbol != 'C':
            raise ValueError(f"Expected carbon atom for C1, got {c1_atom.element_symbol}")
        if o1_atom.element_symbol != 'O':
            raise ValueError(f"Expected oxygen atom for O1, got {o1_atom.element_symbol}")
        if c2_atom.element_symbol != 'C':
            raise ValueError(f"Expected carbon atom for C2, got {c2_atom.element_symbol}")
        
        # Remove Amber cap atoms
        if remove_amber_str == "head":
            final_structure = self._remove_amber_head(final_structure, 
                                                      remake=False,
                                                      )
        elif remove_amber_str == "tail":
            final_structure = self._remove_amber_tail(final_structure, 
                                                      remake=False,
                                                      )
        elif remove_amber_str == "caps":
            final_structure = self._remove_amber_caps(final_structure, 
                                                      remake=False,
                                                      )
        elif remove_amber_str is None: # no removal
            pass
        else:
            raise ValueError(f"Invalid remove_amber_str: {remove_amber_str}")
        
        # Remove hydrogen atoms bonded to the nitrogen and carbon atoms
        final_structure = self._remove_hydrogens_from_atoms(final_structure, 
                                                            [n1_idx, o1_idx], 
                                                            remake=False)
        
        # Create an amide bond between the nitrogen and carbon atoms
        # For amide formation, we need to:
        # 1. Remove the hydroxyl group (OH) from the carboxyl group (water elimination)
        # 2. Create the amide bond between N1 and C1
        
        # Remove the hydroxyl group (OH) from the carboxyl - find and remove the O1 atom
        # First, remove any bonds involving O1 
        bonds_to_remove = []
        for bond in final_structure.bonds:
            if o1_atom in (bond.atom1, bond.atom2):
                bonds_to_remove.append(bond)
        
        # Remove the bonds involving O1
        for bond in bonds_to_remove:
            if bond in final_structure.bonds:
                final_structure.bonds.remove(bond)
        
        # Remove the O1 atom itself (water elimination)
        final_structure.atoms.pop(o1_idx)
        
        # Create the amide bond between N1 and C1 (remake=False to preserve atom references)
        amide_bond = pmd.Bond(n1_atom, c1_atom)
        final_structure.bonds.append(amide_bond)
        
        # Now remake the structure to finalize all changes
        final_structure.remake()
        
        return final_structure

'''