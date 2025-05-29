"""
Ramachandran plot visualization functionality for protein structures.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from typing import Union, Tuple, Optional
from pathlib import Path

def ramachandran_plot(
    coordinates: Union[torch.Tensor, np.ndarray],
    pdb_file: Union[str, Path],
    frame_idx: Optional[int] = None,
    title: str = "Ramachandran Plot",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    alpha: float = 0.6,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a Ramachandran plot using MDAnalysis built-in Ramachandran analysis.
    Plots histogram of phi/psi angles averaged over residues for each frame.
    """
    try:
        from MDAnalysis.analysis.dihedrals import Ramachandran
    except ImportError:
        print("MDAnalysis Ramachandran analysis not available, using custom implementation")
        return ramachandran_plot(coordinates, pdb_file, frame_idx, title, save_path, 
                               dpi, figsize, cmap, alpha, show)
    
    # Convert torch tensor to numpy if needed
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    # Load universe
    u = mda.Universe(str(pdb_file))
    
    # Add coordinates as in-memory trajectory
    coords_A = coordinates.copy()
    # If coordinates are in nm, convert to Angstrom (uncomment if needed)
    # coords_A *= 10.0
    
    u.load_new(coords_A, format=MemoryReader)
    
    # Run Ramachandran analysis
    rama = Ramachandran(u.select_atoms("protein")).run()
    
    # Get angles - shape: [n_frames, n_residues, 2] where 2 = [phi, psi]
    angles = rama.results.angles
    print(f"Original angles shape: {angles.shape}")
    
    # Average over the residues dimension (axis=1) to get [n_frames, 2]
    # Handle NaN values by using nanmean
    phi_per_frame = np.nanmean(angles[:, :, 0], axis=1)  # Shape: [n_frames]
    psi_per_frame = np.nanmean(angles[:, :, 1], axis=1)  # Shape: [n_frames]
    
    print(f"Phi per frame shape: {phi_per_frame.shape}")
    print(f"Psi per frame shape: {psi_per_frame.shape}")
    
    # Remove frames where average is NaN
    valid_mask = ~(np.isnan(phi_per_frame) | np.isnan(psi_per_frame))
    phi_clean = phi_per_frame[valid_mask]
    psi_clean = psi_per_frame[valid_mask]
    
    if len(phi_clean) == 0:
        raise ValueError("No valid phi/psi angles found after averaging.")
    
    print(f"Valid frames after NaN removal: {len(phi_clean)}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create 2D histogram with more bins since we have many more points
    hist, xedges, yedges = np.histogram2d(phi_clean, psi_clean, 
                                         bins=100, range=[[-180, 180], [-180, 180]])
    
    if np.max(hist) > 0:
        # Plot histogram as heatmap with viridis colormap
        # Make zero-count bins transparent by masking them
        hist_masked = np.ma.masked_where(hist == 0, hist)
        im = ax.imshow(hist_masked.T, origin='lower', extent=[-180, 180, -180, 180], 
                       cmap='viridis', aspect='equal')
        # Make colorbar same height as plot area
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Frequency', fontsize=10)
    
    # Remove scatter plot overlay to show only the histogram
    
    # Styling
    ax.set_xlabel('Phi (degrees)', fontsize=10)
    ax.set_ylabel('Psi (degrees)', fontsize=10)
    ax.set_title(f"{title} (Averaged over residues, {len(phi_clean)} frames)", 
                fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    # No reference regions plotted
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    
    # Statistics with compact box
    ax.text(0.02, 0.98, f'N frames: {len(phi_clean)}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax