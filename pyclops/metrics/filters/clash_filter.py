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
import rdkit.Chem as Chem
from typing import Union, Tuple
import tempfile
import os

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

    def get_valid_mask(self, coordinates: TensorLike) -> np.ndarray:
        """Get boolean mask of valid (non-clashing) samples.

        Args:
            coordinates: Sample coordinates [n_batch, n_atoms, 3].

        Returns:
            Boolean array [n_batch] where True = valid (no clashes).
        """
        coords = self._preprocess_coordinates(coordinates)
        distances = self._calculate_pairwise_distances(coords)
        return self._compute_good_sample_mask(distances)

    def filter_and_mask(self, coordinates: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out samples with heavy atom clashes and return a mask of valid samples.
        """
        coords = self._preprocess_coordinates(coordinates)
        distances = self._calculate_pairwise_distances(coords)
        good_sample_mask = self._compute_good_sample_mask(distances)
        return coords[good_sample_mask], good_sample_mask

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

    @classmethod
    def from_rdkit_mol(
        cls,
        mol: Chem.Mol,
        units_factor: float = 10.0,
        clash_cutoff: float = 0.63,
        include_hydrogens: bool = False,
        except_disulfide_bonds: bool = True,
        except_bonded_pairs: bool = True,
    ) -> "ClashFilter":
        """Create ClashFilter from an RDKit molecule.
        
        The molecule must have a conformer with 3D coordinates. Use 
        rdkit.Chem.AllChem.EmbedMolecule() to generate coordinates if needed.
        
        Args:
            mol: RDKit molecule with at least one conformer containing 3D coordinates.
            units_factor: Factor to convert input coordinates to Angstroms.
            clash_cutoff: Fraction of sum of radii below which atoms are clashing.
            include_hydrogens: Whether to include hydrogen atoms in clash detection.
            except_disulfide_bonds: Whether to exclude disulfide bonds from clash detection.
            except_bonded_pairs: Whether to exclude bonded atom pairs from clash detection.
            
        Returns:
            ClashFilter instance configured for the molecule's topology.
            
        Raises:
            ValueError: If the molecule has no conformers.
        """
        if mol.GetNumConformers() == 0:
            raise ValueError(
                "RDKit molecule has no conformers. Use AllChem.EmbedMolecule() "
                "to generate 3D coordinates first."
            )
        
        # Convert RDKit mol to PDB block, then load via mdtraj
        pdb_block = Chem.MolToPDBBlock(mol)
        
        # Write to temp file and load with mdtraj (mdtraj doesn't support StringIO)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_block)
            temp_path = f.name
        
        topology = md.load_pdb(temp_path).topology
        os.unlink(temp_path)
        
        return cls(
            topology, 
            units_factor, 
            clash_cutoff, 
            include_hydrogens, 
            except_disulfide_bonds, 
            except_bonded_pairs
        )