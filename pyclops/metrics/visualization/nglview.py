"""
Interactive protein structure visualization using nglview.
"""
import numpy as np
import torch
import nglview as nv
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from typing import Union, Optional
from pathlib import Path

def visualize_structure( # add units_factor
    coordinates: Union[torch.Tensor, np.ndarray],
    units_factor: float,
    pdb_file: Union[str, Path],
    frame_idx: Optional[int] = None,
    style: str = "cartoon",
    color_scheme: str = "residueindex",
    background_color: str = "white",
    show_atoms: bool = False,
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
    u = mda.Universe(str(pdb_file))
    
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
