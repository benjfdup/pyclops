"""
Ramachandran plot visualization functionality for protein structures.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import MDAnalysis as mda
from typing import Union, Tuple, Optional
from pathlib import Path

def ramachandran_plot(
    coordinates: Union[torch.Tensor, np.ndarray],
    pdb_file: Union[str, Path],
    frame_idx: int = 0,
    title: str = "Ramachandran Plot",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    alpha: float = 0.6,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a Ramachandran plot from protein coordinates.

    Parameters
    ----------
    coordinates : Union[torch.Tensor, np.ndarray]
        Protein coordinates of shape [n_frames, n_atoms, 3]
    pdb_file : Union[str, Path]
        Path to the PDB file containing the protein topology
    frame_idx : int, optional
        Index of the frame to plot, by default 0
    title : str, optional
        Title for the plot, by default "Ramachandran Plot"
    save_path : Optional[Union[str, Path]], optional
        Path to save the plot, by default None
    dpi : int, optional
        DPI for the saved figure, by default 300
    figsize : Tuple[int, int], optional
        Figure size in inches, by default (10, 10)
    cmap : str, optional
        Colormap for the density plot, by default "viridis"
    alpha : float, optional
        Transparency of the scatter points, by default 0.6
    show : bool, optional
        Whether to display the plot, by default True

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The figure and axes objects of the plot
    """
    # Convert torch tensor to numpy if necessary
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()
    
    # Load the protein structure
    u = mda.Universe(pdb_file)
    
    # Select protein atoms
    protein = u.select_atoms("protein")
    
    # Verify coordinate dimensions match
    if coordinates.shape[1] != len(protein.atoms):
        raise ValueError(
            f"Number of atoms in coordinates ({coordinates.shape[1]}) does not match "
            f"number of atoms in PDB ({len(protein.atoms)})"
        )
    
    # Get the frame coordinates
    frame_coords = coordinates[frame_idx]
    
    # Update coordinates in the universe
    protein.positions = frame_coords
    
    # Calculate phi and psi angles
    protein_angles = protein.angles
    phi_angles = protein_angles.phi_angles()
    psi_angles = protein_angles.psi_angles()
    
    # Remove any NaN values
    mask = ~(np.isnan(phi_angles) | np.isnan(psi_angles))
    phi_angles = phi_angles[mask]
    psi_angles = psi_angles[mask]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(
        phi_angles, psi_angles,
        bins=50,
        range=[[-180, 180], [-180, 180]]
    )
    
    # Plot the density
    im = ax.imshow(
        h.T,
        origin='lower',
        extent=[-180, 180, -180, 180],
        aspect='auto',
        cmap=cmap,
        alpha=alpha
    )
    
    # Add scatter plot of individual points
    ax.scatter(
        phi_angles,
        psi_angles,
        c='black',
        s=10,
        alpha=alpha/2,
        marker='.'
    )
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Density')
    
    # Set labels and title
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add common secondary structure regions
    regions = {
        'α-helix': (-60, -45),
        'β-sheet': (-120, 120),
        'Left-handed α-helix': (45, 60)
    }
    
    for name, (phi, psi) in regions.items():
        ax.axvline(x=phi, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=psi, color='red', linestyle='--', alpha=0.3)
    
    # Save the plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return fig, ax 