import numpy as np
import torch
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from typing import Union, Tuple, Optional, List
from pathlib import Path
from sklearn.decomposition import PCA

def torsional_pca_plot(
    coordinates: Union[torch.Tensor, np.ndarray],
    pdb_file: Union[str, Path],
    frame_idx: Optional[int] = None,
    title: str = "Torsional PCA Plot",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    alpha: float = 0.6,
    show: bool = True,
    pc_vectors: Optional[List[Union[torch.Tensor, np.ndarray]]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a torsional PCA plot using MDAnalysis built-in Ramachandran analysis.
    Performs PCA on flattened phi/psi angles and plots the distribution of the first two principal components.
    
    Args:
        coordinates: Input coordinates
        pdb_file: Path to PDB file
        frame_idx: Optional frame index
        title: Plot title
        save_path: Optional path to save the plot
        dpi: DPI for the plot
        figsize: Figure size
        cmap: Colormap for the histogram
        alpha: Alpha value for the plot
        show: Whether to show the plot
        pc_vectors: Optional list of two PC vectors to project the data onto. If provided, 
                   these vectors will be used instead of computing new PCs.
    """
    try:
        from MDAnalysis.analysis.dihedrals import Ramachandran
    except ImportError:
        raise ImportError("MDAnalysis Ramachandran analysis not available")
    
    # Convert torch tensor to numpy if needed
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.cpu().numpy()
    
    # Load universe
    u = mda.Universe(str(pdb_file))
    
    # Add coordinates as in-memory trajectory
    coords_A = coordinates.copy()
    u.load_new(coords_A, format=MemoryReader)
    
    # Run Ramachandran analysis
    rama = Ramachandran(u.select_atoms("protein")).run()
    
    # Get angles - shape: [n_frames, n_residues, 2] where 2 = [phi, psi]
    angles = rama.results.angles
    print(f"Original angles shape: {angles.shape}")
    
    # Flatten the angles array to [n_frames, n_residues * 2]
    n_frames, n_residues, _ = angles.shape
    flattened_angles = angles.reshape(n_frames, n_residues * 2)
    
    # Remove any frames that have NaN values
    valid_mask = ~np.isnan(flattened_angles).any(axis=1)
    flattened_angles_clean = flattened_angles[valid_mask]
    
    if len(flattened_angles_clean) == 0:
        raise ValueError("No valid frames found after removing NaN values.")
    
    print(f"Valid frames after NaN removal: {len(flattened_angles_clean)}")
    
    # Handle PC vectors if provided
    if pc_vectors is not None:
        if len(pc_vectors) != 2:
            raise ValueError("Exactly two PC vectors must be provided")
        
        # Convert PC vectors to numpy if they're torch tensors
        pc_vectors_np = [vec.cpu().numpy() if isinstance(vec, torch.Tensor) else vec 
                        for vec in pc_vectors]
        
        # Stack vectors into a projection matrix
        projection_matrix = np.vstack(pc_vectors_np)
        
        # Project the data
        pca_result = flattened_angles_clean @ projection_matrix.T
        
        # Calculate explained variance manually
        centered_data = flattened_angles_clean - np.mean(flattened_angles_clean, axis=0)
        total_var = np.sum(np.var(centered_data, axis=0))
        pc1_var = np.var(pca_result[:, 0]) / total_var
        pc2_var = np.var(pca_result[:, 1]) / total_var
        explained_variance_ratio = [pc1_var, pc2_var]
        
    else:
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(flattened_angles_clean)
        explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create 2D histogram of PCA components
    hist, xedges, yedges = np.histogram2d(pca_result[:, 0], pca_result[:, 1], 
                                         bins=100)
    
    if np.max(hist) > 0:
        # Plot histogram as heatmap with viridis colormap
        hist_masked = np.ma.masked_where(hist == 0, hist)
        im = ax.imshow(hist_masked.T, origin='lower', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                      cmap='viridis', aspect='equal')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Frequency', fontsize=10)
    
    # Styling
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)', fontsize=10)
    
    # Update title based on whether custom PCs were used
    pc_source = "Custom PC" if pc_vectors is not None else "PCA"
    ax.set_title(f"{title} ({pc_source} of {n_residues} residues, {len(flattened_angles_clean)} frames)", 
                fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Statistics with compact box
    ax.text(0.02, 0.98, f'N frames: {len(flattened_angles_clean)}\nN residues: {n_residues}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax