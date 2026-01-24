"""
Heavy atom clash detection and filtering for molecular samples.

Two atoms "clash" when they're closer than some fraction (default 0.63) of the
sum of their van der Waals radii. This indicates steric overlap that would be
physically unrealistic.
"""
# https://www.blopig.com/blog/2023/05/checking-your-pdb-file-for-clashing-atoms/
__all__ = ["ClashFilter"]

import numpy as np
import torch
import mdtraj as md
from typing import Union, Tuple

TensorLike = Union[torch.Tensor, np.ndarray]


class ClashFilter:
    """Filter molecular samples based on heavy atom steric clashes."""

    # Van der Waals radii in Angstroms (from standard tables)
    _ATOM_RADII = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "F": 1.47,
        "P": 1.80,
        "CL": 1.75,
        "MG": 1.73,
    }

    # Disulfide bonds (S-S) are ~2.05A, so anything >1.88A is a valid bond, not a clash
    _DISULFIDE_CUTOFF = 1.88  # Angstroms

    def __init__(
        self,
        topology: md.Topology,
        units_factor: float = 10.0,  # input_coords * units_factor = Angstroms
        clash_cutoff: float = 0.63,  # fraction of sum of radii
        include_hydrogens: bool = False, # hydrogen clashes are less meaningful and hydrogens are often not present in structures
        except_disulfide_bonds: bool = True, # S-S bonds are handled separately since they need distance-based logic (close = bond, far = clash)
        except_bonded_pairs: bool = True, # pairs of atoms that are bonded to each other are excluded from the clash detection

    ):
        """
        Args:
            topology: MDTraj topology for the system.
            units_factor: Factor to convert input coordinates to Angstroms.
            clash_cutoff: Fraction of sum of radii below which atoms are clashing.
        """
        self._topology = topology
        self._units_factor = units_factor
        self._clash_cutoff = clash_cutoff

        self._include_hydrogens = include_hydrogens
        self._except_disulfide_bonds = except_disulfide_bonds
        self._except_bonded_pairs = except_bonded_pairs

        self._clash_thresholds = self._compute_clash_thresholds()

    # FINISH IMPLEMENTING THIS FUNCTION
    def _compute_clash_thresholds(self) -> np.ndarray:
        """Compute pairwise clash distance thresholds for all heavy atom pairs.
        
        For each pair of atoms i,j: threshold = clash_cutoff * (radius_i + radius_j)
        If atoms are closer than this, they're clashing.
        """
        # returns a matrix of shape (n_atoms, n_atoms) where each entry is the clash threshold
        # for the pair of the atoms at the indices of the matrix.
        # an entry of 0.0 means that the atom is excluded from the clash detection 
        # (since all distances will be greater than that threshold)

        # make 
        
        n_atoms = self._topology.n_atoms
        
        # Build radii array: 0.0 for excluded atoms (unknown elements, or H if not included)
        radii = np.zeros(n_atoms, dtype=np.float64)
        for atom in self._topology.atoms:
            elem = atom.element.symbol.upper()
            if elem not in self._ATOM_RADII:
                #continue
                raise ValueError(f"Atom {atom.name} has unknown element {elem}")
            if elem == "H" and not self._include_hydrogens:
                continue
            radii[atom.index] = self._ATOM_RADII[elem]

        # Vectorized outer sum: thresholds[i,j] = clash_cutoff * (radii[i] + radii[j])
        # If either radius is 0, threshold is 0 (excluded from detection)
        thresholds = self._clash_cutoff * (radii[:, np.newaxis] + radii[np.newaxis, :])

        # Exclude disulfide bonds (SG-SG pairs) from clash detection
        if self._except_disulfide_bonds:
            sg_indices = [atom.index for atom in self._topology.atoms if atom.name == "SG"]
            for i in sg_indices:
                for j in sg_indices:
                    thresholds[i, j] = self._DISULFIDE_CUTOFF

        # Exclude bonded atom pairs from clash detection
        if self._except_bonded_pairs:
            for bond in self._topology.bonds:
                i, j = bond[0].index, bond[1].index
                thresholds[i, j] = 0.0
                thresholds[j, i] = 0.0

        # Exclude self-interactions (diagonal)
        np.fill_diagonal(thresholds, 0.0)

        return thresholds

    def _calculate_pairwise_distances(self, coords: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between all atoms in the topology."""
        # coords is a numpy array of shape (n_batch, n_atoms, 3) or (n_atoms, 3) in Angstroms
        # return a numpy array of shape (n_batch, n_atoms, n_atoms)
        
        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]  # (n_atoms, 3) -> (1, n_atoms, 3)
        
        # diff[b, i, j, :] = coords[b, i, :] - coords[b, j, :]
        diff = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _compute_good_sample_mask(self, distances: np.ndarray) -> np.ndarray:
        """Compute a mask of good samples based on the distances between all atoms."""
        # distances is a numpy array of shape (n_batch, n_atoms, n_atoms)
        # return a numpy array of shape (n_batch, )
        
        # good samples have NO clashes (all distances >= threshold)
        return ~np.any(distances < self._clash_thresholds, axis=(1, 2))

    def _preprocess_coordinates(self, coordinates: TensorLike) -> np.ndarray:
        """Preprocess the coordinates to be in Angstroms."""
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.detach().cpu().numpy()
        else:
            coords = np.asarray(coordinates)

        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]  # (n_atoms, 3) -> (1, n_atoms, 3)

        if coords.shape[1] != self._topology.n_atoms:
            raise ValueError(f"Number of atoms in coordinates ({coords.shape[1]}) doesn't match topology ({self._topology.n_atoms})")

        if coords.shape[2] != 3:
            raise ValueError(f"Coordinates must have 3 dimensions, got {coords.shape[2]}")
        
        return coords * self._units_factor

    def filter(self, coordinates: TensorLike) -> np.ndarray:
        """Filter out samples with heavy atom clashes.

        ALWAYS returns a numpy array of shape (n_batch, n_atoms, 3) with units in Angstroms
        """

        coords = self._preprocess_coordinates(coordinates)
        distances = self._calculate_pairwise_distances(coords)

        good_sample_mask = self._compute_good_sample_mask(distances)
        # always returns a numpy array of shape (n_batch, n_atoms, 3) with units in Angstroms
        return coords[good_sample_mask]

    def count_clashes(self, coordinates: TensorLike) -> np.ndarray:
        """Count the number of clashes in a sample.
        Returns a numpy array of shape (n_batch, ) with the number of clashes per sample.
        """
        coords = self._preprocess_coordinates(coordinates)
        distances = self._calculate_pairwise_distances(coords)
        return np.sum(distances < self._clash_thresholds, axis=(1, 2))//2 # divide by 2 because each clash is counted twice

    @classmethod
    def from_pdb_file(
        cls,
        pdb_file: str,
        units_factor: float = 10.0,  # nanometers to angstroms
        clash_cutoff: float = 0.63,
        include_hydrogens: bool = False, # hydrogen clashes are less meaningful and hydrogens are often not present in structures
        except_disulfide_bonds: bool = True, # S-S bonds are handled separately since they need distance-based logic (close = bond, far = clash)
        except_bonded_pairs: bool = True, # pairs of atoms that are bonded to each other are excluded from the clash detection
        
    ) -> "ClashFilter":
        """Create ClashFilter from a PDB file."""
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor, clash_cutoff, include_hydrogens, except_disulfide_bonds, except_bonded_pairs)
        
        # ================================ OLD CODE ================================
        '''
        # Pre-compute everything at init so clash detection is fast
        # We store: which atoms to check, their elements/names/residues, 
        # the distance thresholds, and which pairs to skip
        self._heavy_indices, self._elements, self._atom_names, self._residue_indices = (
            self._extract_atom_info()
        )
        self._clash_thresholds = self._compute_clash_thresholds()
        self._exclusion_mask = self._build_exclusion_mask()

    def _extract_atom_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract heavy atom indices and their properties from topology.
        
        We only care about heavy atoms (non-H) since hydrogen clashes are 
        less meaningful and hydrogens are often not present in structures.
        """
        heavy_indices = []
        elements = []
        atom_names = []
        residue_indices = []

        for atom in self._topology.atoms:
            elem = atom.element.symbol.upper()
            # Only keep atoms we have radii for, excluding hydrogen
            if elem in self._ATOM_RADII and elem != "H":
                heavy_indices.append(atom.index)
                elements.append(elem)
                atom_names.append(atom.name)  # e.g., "CA", "CB", "SG", "N", "C"
                residue_indices.append(atom.residue.index)

        return (
            np.array(heavy_indices),
            np.array(elements),
            np.array(atom_names),
            np.array(residue_indices),
        )

    def _compute_clash_thresholds(self) -> np.ndarray:
        """Compute pairwise clash distance thresholds for all heavy atom pairs.
        
        For each pair of atoms i,j: threshold = clash_cutoff * (radius_i + radius_j)
        If atoms are closer than this, they're clashing.
        """
        n_heavy = len(self._heavy_indices)
        thresholds = np.zeros((n_heavy, n_heavy), dtype=np.float64)

        for i, elem_i in enumerate(self._elements):
            for j, elem_j in enumerate(self._elements):
                radius_sum = self._ATOM_RADII[elem_i] + self._ATOM_RADII[elem_j]
                thresholds[i, j] = self._clash_cutoff * radius_sum

        return thresholds

    def _build_exclusion_mask(self) -> np.ndarray:
        """Build mask for atom pairs to exclude from clash detection.

        We exclude pairs that are SUPPOSED to be close:
            - Same residue: atoms within a residue are bonded/close by design
            - Peptide bonds: backbone C-N between adjacent residues
        
        Disulfide bridges (SG-SG) are handled separately since they need
        distance-based logic (close = bond, far = clash).
        """
        n_heavy = len(self._heavy_indices)
        # True = exclude this pair from clash detection
        exclusion = np.zeros((n_heavy, n_heavy), dtype=bool)

        for i in range(n_heavy):
            for j in range(n_heavy):
                # Always exclude self-comparison
                if i == j:
                    exclusion[i, j] = True
                    continue

                # Same residue = bonded or 1-3 interactions, not clashes
                if self._residue_indices[i] == self._residue_indices[j]:
                    exclusion[i, j] = True
                    continue

                # Peptide bond: backbone C of residue i bonds to N of residue i+1
                name_i, name_j = self._atom_names[i], self._atom_names[j]
                res_i, res_j = self._residue_indices[i], self._residue_indices[j]
                if abs(res_i - res_j) == 1:
                    if (name_i == "C" and name_j == "N") or (name_i == "N" and name_j == "C"):
                        exclusion[i, j] = True

        return exclusion

    def _is_disulfide_pair(self, i: int, j: int) -> bool:
        """Check if atom pair is a potential disulfide bridge (SG-SG).
        
        SG = sulfur gamma, the sulfur in cysteine sidechains that forms S-S bonds.
        """
        return self._atom_names[i] == "SG" and self._atom_names[j] == "SG"

    def _get_heavy_atom_indices(self) -> np.ndarray:
        """Extract indices of heavy (non-hydrogen) atoms from topology."""
        return self._heavy_indices

    def _compute_pairwise_distances(
        self,
        coords: np.ndarray,  # [n_batch, n_heavy_atoms, 3] in Angstroms
    ) -> np.ndarray:
        """
        Compute pairwise distances between heavy atoms.

        Args:
            coords: Heavy atom coordinates [n_batch, n_heavy_atoms, 3].

        Returns:
            Pairwise distance matrix [n_batch, n_heavy_atoms, n_heavy_atoms].
        """
        # Broadcasting trick: expand coords to [batch, atoms, 1, 3] and [batch, 1, atoms, 3]
        # then subtract to get [batch, atoms, atoms, 3] of displacement vectors
        diff = coords[:, :, np.newaxis, :] - coords[:, np.newaxis, :, :]
        # Euclidean distance along last axis
        return np.sqrt(np.sum(diff**2, axis=-1))

    def detect_clashes(
        self,
        coordinates: TensorLike,  # [n_batch, n_atoms, 3]
    ) -> np.ndarray:
        """
        Detect which samples have heavy atom clashes.

        Args:
            coordinates: Sample coordinates [n_batch, n_atoms, 3].

        Returns:
            Boolean array [n_batch] where True indicates a clash.
        """
        # Handle both torch and numpy inputs
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.detach().cpu().numpy()
        else:
            coords = np.asarray(coordinates)

        # Convert from input units (likely nm) to Angstroms for comparison with radii
        coords = coords * self._units_factor

        # Pull out just the heavy atoms we care about
        heavy_coords = coords[:, self._heavy_indices, :]

        # Get all pairwise distances in one vectorized call
        distances = self._compute_pairwise_distances(heavy_coords)

        # Check each sample for clashes
        n_batch = coords.shape[0]
        has_clash = np.zeros(n_batch, dtype=bool)

        for b in range(n_batch):
            dist_b = distances[b]
            # Only check upper triangle (i < j) to avoid counting each clash twice
            for i in range(len(self._heavy_indices)):
                for j in range(i + 1, len(self._heavy_indices)):
                    # Skip excluded pairs (same residue, peptide bonds)
                    if self._exclusion_mask[i, j]:
                        continue

                    # Special case: SG-SG pairs might be disulfide bonds
                    # If they're farther than 1.88A, it's a valid bond, not a clash
                    if self._is_disulfide_pair(i, j) and dist_b[i, j] > self._DISULFIDE_CUTOFF:
                        continue

                    # Finally: is the distance below the clash threshold?
                    if dist_b[i, j] < self._clash_thresholds[i, j]:
                        has_clash[b] = True
                        break  # One clash is enough, skip to next sample
                if has_clash[b]:
                    break

        return has_clash

    def filter(
        self,
        coordinates: TensorLike,  # [n_batch, n_atoms, 3]
    ) -> TensorLike:
        """
        Filter out samples with heavy atom clashes.

        Args:
            coordinates: Sample coordinates [n_batch, n_atoms, 3].

        Returns:
            Filtered coordinates with clashing samples removed.
        """
        valid_mask = self.get_valid_mask(coordinates)

        # Preserve input type (torch or numpy)
        if isinstance(coordinates, torch.Tensor):
            return coordinates[torch.from_numpy(valid_mask)]
        return coordinates[valid_mask]

    def get_valid_mask(
        self,
        coordinates: TensorLike,  # [n_batch, n_atoms, 3]
    ) -> np.ndarray:
        """
        Get boolean mask of valid (non-clashing) samples.

        Args:
            coordinates: Sample coordinates [n_batch, n_atoms, 3].

        Returns:
            Boolean array [n_batch] where True indicates valid sample.
        """
        # Valid = NOT clashing
        return ~self.detect_clashes(coordinates)

    def count_clashes(
        self,
        coordinates: TensorLike,  # [n_batch, n_atoms, 3]
    ) -> np.ndarray:
        """
        Count number of clashes per sample.

        Useful for analysis/debugging to see how "bad" each sample is.

        Args:
            coordinates: Sample coordinates [n_batch, n_atoms, 3].

        Returns:
            Array [n_batch] with clash counts per sample.
        """
        # Same setup as detect_clashes...
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.detach().cpu().numpy()
        else:
            coords = np.asarray(coordinates)

        coords = coords * self._units_factor
        heavy_coords = coords[:, self._heavy_indices, :]
        distances = self._compute_pairwise_distances(heavy_coords)

        n_batch = coords.shape[0]
        clash_counts = np.zeros(n_batch, dtype=int)

        # ...but count ALL clashes instead of stopping at first
        for b in range(n_batch):
            dist_b = distances[b]
            for i in range(len(self._heavy_indices)):
                for j in range(i + 1, len(self._heavy_indices)):
                    if self._exclusion_mask[i, j]:
                        continue

                    if self._is_disulfide_pair(i, j) and dist_b[i, j] > self._DISULFIDE_CUTOFF:
                        continue

                    if dist_b[i, j] < self._clash_thresholds[i, j]:
                        clash_counts[b] += 1

        return clash_counts

    @classmethod
    def from_pdb_file(
        cls,
        pdb_file: str,
        units_factor: float = 10.0,  # nanometers to angstroms
        clash_cutoff: float = 0.63,
    ) -> "ClashFilter":
        """Create ClashFilter from a PDB file."""
        topology = md.load_pdb(pdb_file).topology
        return cls(topology, units_factor, clash_cutoff)

    '''