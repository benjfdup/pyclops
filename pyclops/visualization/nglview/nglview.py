"""
Interactive protein structure visualization using nglview.
"""
__all__ = [
    "NGLViewVisualizer",
]

import numpy as np
import torch
import nglview as nv
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from typing import Union, Optional
from pathlib import Path

from ...core.chemical_loss.chemical_loss import ChemicalLoss

# type aliases
TensorLike = Union[torch.Tensor, np.ndarray]
PathLike = Union[str, Path]

class NGLViewVisualizer:
    """
    A class for visualizing protein structures and chemical losses.
    """
    def __init__(self,
                 pdb_file: PathLike,
                 ):
        self._pdb_file = pdb_file

    def visualize_structure(
        self,
        coordinates: TensorLike,
        units_factor: float,
        frame_idx: Optional[int] = None,
        style: str = "cartoon",
        color_scheme: str = "residueindex",
        background_color: str = "white",
        show_atoms: bool = True,
        atom_scale: float = 0.5,
        atom_opacity: float = 0.6
    ) -> nv.NGLWidget:
        """
        Create an interactive nglview visualization of a protein structure.
        
        Args:
            coordinates: Array of shape [n_frames, n_atoms, 3] containing atomic coordinates
            pdb_file: Path to the PDB file containing topology information
            frame_idx: Optional frame index to display (if None, shows first frame)
            style: Visualization style (e.g., 'cartoon', 'line', 'ball+stick')
            color_scheme: Color scheme for the visualization
            background_color: Background color of the viewer
            show_atoms: Whether to show atomic representation alongside the main style
            atom_scale: Scale factor for atomic spheres (default: 0.5)
            atom_opacity: Opacity of atomic spheres (default: 0.6)
            
        Returns:
            nglview.NGLWidget: Interactive visualization widget
        """
        # Convert torch tensor to numpy if needed
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        
        # Load universe
        u = mda.Universe(str(self._pdb_file))
        
        # Add coordinates as in-memory trajectory
        coords_A = coordinates.copy()
        coords_A *= units_factor # convert to Angstroms (which is what MDAnalysis expects)
        
        u.load_new(coords_A, format=MemoryReader)
        
        # Create nglview widget
        view = nv.show_mdanalysis(u)
        
        # Set visualization parameters
        view.clear_representations()
        
        # Add main representation (cartoon/ribbon)
        view.add_representation(style, color_scheme=color_scheme)
        
        # Add atomic representation if requested
        if show_atoms:
            view.add_representation('ball+stick', 
                                color_scheme=color_scheme,
                                scale=atom_scale,
                                opacity=atom_opacity)
        
        view.background = background_color
        
        # Set frame if specified
        if frame_idx is not None:
            view.frame = frame_idx
        
        return view


    def visualize_loss_atoms(
        self,
        coordinates: TensorLike,
        chemical_loss: ChemicalLoss,
        units_factor: float,
        frame_idx: Optional[int] = None,
        style: str = "cartoon",
        color_scheme: str = "residueindex",
        background_color: str = "white",
        show_atoms: bool = True,
        atom_scale: float = 0.5,
        atom_opacity: float = 0.6,
        highlight_color: str = "magenta",
        highlight_scale: float = 1.2,
        highlight_opacity: float = 0.9
    ) -> nv.NGLWidget:
        """
        Create an interactive nglview visualization of the atoms considered in a chemical loss.
        
        This function creates a visualization similar to visualize_structure but highlights
        the specific atoms involved in the chemical loss (the 4 vertices of the tetrahedron).
        
        Args:
            coordinates: Array of shape [n_frames, n_atoms, 3] containing atomic coordinates
            chemical_loss: ChemicalLoss instance containing the atom indices to highlight
            units_factor: Factor to convert coordinates to Angstroms
            pdb_file: Path to the PDB file containing topology information
            frame_idx: Optional frame index to display (if None, shows first frame)
            style: Visualization style for the main structure (e.g., 'cartoon', 'line', 'ball+stick')
            color_scheme: Color scheme for the main visualization
            background_color: Background color of the viewer
            show_atoms: Whether to show atomic representation alongside the main style
            atom_scale: Scale factor for regular atomic spheres (default: 0.5)
            atom_opacity: Opacity of regular atomic spheres (default: 0.6)
            highlight_color: Color for the highlighted atoms involved in the chemical loss
            highlight_scale: Scale factor for the highlighted atoms (default: 1.2)
            highlight_opacity: Opacity of the highlighted atoms (default: 0.9)
            
        Returns:
            nglview.NGLWidget: Interactive visualization widget with highlighted loss atoms
        """
        # Convert torch tensor to numpy if needed
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        
        # Load universe
        u = mda.Universe(str(self._pdb_file))
        
        # Add coordinates as in-memory trajectory
        coords_A = coordinates.copy()
        coords_A *= units_factor # convert to Angstroms (which is what MDAnalysis expects)
        
        u.load_new(coords_A, format=MemoryReader)
        
        # Create nglview widget
        view = nv.show_mdanalysis(u)
        
        # Set visualization parameters
        view.clear_representations()
        
        # Add main representation (cartoon/ribbon)
        view.add_representation(style, color_scheme=color_scheme)
        
        # Add atomic representation if requested
        if show_atoms:
            view.add_representation('ball+stick', 
                                color_scheme=color_scheme,
                                scale=atom_scale,
                                opacity=atom_opacity)
        
        # Get the atom indices involved in the chemical loss
        # chemical_loss.vertex_indices is a tensor of shape [4] with global atom indices
        loss_atom_indices = chemical_loss.vertex_indices.cpu().numpy()
        
        # Create selection string for the highlighted atoms
        # MDAnalysis uses 0-based indexing, so we need to add 1 for the selection string
        atom_selection = " or ".join([f"index {idx}" for idx in loss_atom_indices])
        
        # Add highlighted representation for the loss atoms
        view.add_representation('ball+stick',
                            selection=atom_selection,
                            color=highlight_color,
                            scale=highlight_scale,
                            opacity=highlight_opacity)
        
        # Also add a spacefill representation to make them even more visible
        view.add_representation('spacefill',
                            selection=atom_selection,
                            color=highlight_color,
                            scale=highlight_scale * 0.8,
                            opacity=highlight_opacity * 0.7)
        
        view.background = background_color
        
        # Set frame if specified
        if frame_idx is not None:
            view.frame = frame_idx
        
        return view
