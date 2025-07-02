import torch

def motif_loss(
        pos: torch.Tensor, # TODO: speed up with torch.jit.script
        ref_pos: torch.Tensor, 
        tolerance: float = 0.0, 
        squared: bool = False,
        allow_flips: bool = False,
        ):
    
    """
    Computes a rotationally and translationally invariant structural deviation loss.

    This function first aligns `pos` to `ref_pos` using the Kabsch algorithm, ensuring 
    rotational and translational invariance. After alignment, it computes the deviation 
    of each atom and applies a soft tolerance, where deviations within `tolerance` are 
    ignored, and those exceeding it are penalized quadratically if squared is true, 
    else linearly.

    Parameters:
    ----------
    pos : torch.tensor
        Tensor of shape [n_batch, n_atoms, 3] representing current particle positions 
        in Angstroms.
    ref_pos : torch.tensor
        Tensor of shape [n_atoms, 3] representing the ideal particle positions in Angstroms.
    tolerance : float, optional
        No-penalty range around the target positions in Angstroms. Deviations within this 
        range do not contribute to the loss. Defaults to 0.0.
    squared : bool, optional
        If True, returns the Mean Squared Deviation (MSD). If False, returns the 
        Root Mean Squared Deviation (RMSD). Defaults to False.
    allow_flips : bool, optional
        If True, allows the Kabsch algorithm to flip the coordinate system. Defaults to False.

    Returns:
    -------
    torch.tensor
        A loss tensor of shape [n_batch, ], where each value represents the deviation 
        of a structure from the reference after optimal alignment.
    """
    assert pos.shape[1:] == ref_pos.shape, "pos must have the same shape as ref_pos, excluding the batch (first) dimension."
    
    # this function is largely written by Hunter Heidenreich.
    # its source can be found here:
    # https://hunterheidenreich.com/posts/kabsch_algorithm/

    P = pos
    Q = ref_pos
    
    # below code by Hunter Heidenreich
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    if allow_flips:
        d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
        flip = d < 0.0
        if flip.any().item():
            Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
    # above code from Hunter Heidenreich

    # Align positions using optimal rotation
    aligned_positions = torch.matmul(p, R.transpose(1, 2))  # [B, N, 3]

    # Compute per-atom squared deviation
    per_atom_sq_dev = torch.sum((aligned_positions - q) ** 2, dim=-1)  # [B, N]
    per_atom_dev = torch.sqrt(per_atom_sq_dev)  # [B, N] - Convert to absolute distances

    # Apply soft tolerance: Ignore deviations ≤ tolerance, quadratic penalty for larger deviations
    error = torch.clamp(per_atom_dev - tolerance, min=0.0)  # [B, N]
    penalty = error ** 2  # Quadratic penalty for smooth gradient

    # Compute mean loss across atoms
    msd = torch.sum(penalty, dim=1) / P.shape[1]  # [B]

    # Compute RMSD or return squared loss
    loss = msd if squared else torch.sqrt(msd)  # [B]

    return loss

@torch.jit.script
def soft_min(inputs: torch.Tensor, alpha: float = -3.0) -> torch.Tensor:
    """
    Numerically stable soft minimum across batches.
    As alpha → -∞, returns hard min. 
    As alpha → 0, returns average.
    As alpha → ∞, returns hard max.
    
    Args:
        inputs: Tensor of shape [batch_size, n_losses]
        alpha: Smoothness parameter (negative for soft min)
        
    Returns:
        Soft minimum across each batch (shape: [batch_size])
    """
    # Subtract max per row for numerical stability
    max_vals = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_vals

    exps = torch.exp(alpha * shifted_inputs)
    weights = exps / exps.sum(dim=-1, keepdim=True)
    result = torch.sum(inputs * weights, dim=-1)

    return result

def soft_max(inputs: torch.Tensor, alpha: float = 3.0) -> torch.Tensor:
    '''
    Numerically stable soft maximum across batches.
    As alpha → -∞, returns hard min. 
    As alpha → 0, returns average.
    As alpha → ∞, returns hard max.
    
    Args:
        inputs: Tensor of shape [batch_size, n_losses]
        alpha: Smoothness parameter (positive for soft max)
    
    Returns:
        Soft maximum across each batch (shape: [batch_size])
    '''
    return soft_min(inputs, alpha = alpha)

def compute_signed_tetrahedral_volume( # will be useful for chirality verification.
    a1: torch.Tensor,  # shape: [n_batch, n_tetrahedron, 3] or [n_batch, 3]
    a2: torch.Tensor,
    a3: torch.Tensor,
    a4: torch.Tensor,
):
    '''
    Computes the signed volume of each tetrahedron in parallel over
    both batch and number of tetrahedra.

    Returns:
    - volumes: tensor of shape [n_batch, n_tetrahedron] or [n_batch] if only 1 tetrahedron is provided.
    '''
    added_dummy_dim = False
    if a1.ndim == 2:  # shape [n_batch, 3]
        a1 = a1.unsqueeze(1)
        a2 = a2.unsqueeze(1)
        a3 = a3.unsqueeze(1)
        a4 = a4.unsqueeze(1)
        added_dummy_dim = True

    v1 = a2 - a1
    v2 = a3 - a1
    v3 = a4 - a1
    cross = torch.cross(v1, v2, dim=-1)
    volume = torch.einsum('bij,bij->bi', cross, v3) / 6.0

    if added_dummy_dim:
        volume = volume.squeeze(1)  # shape: [n_batch]

    return volume

# unsure of where else to put this
def _inherit_docstring(parent_method):
    def decorator(method):
        method.__doc__ = parent_method.__doc__
        return method
    return decorator