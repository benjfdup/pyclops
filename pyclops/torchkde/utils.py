import torch


def ensure_two_dimensional(tensor):
    if tensor.dim() == 0:  # Scalar tensor
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Make it 1x1
    elif tensor.dim() == 1:  # 1D tensor
        tensor = tensor.unsqueeze(0)  # Add a batch dimension
    return tensor


def check_if_mat(matrix_or_scalar):
    if isinstance(matrix_or_scalar, torch.Tensor) and matrix_or_scalar.dim() > 1:
        return True
    else:  # Scalar case
        return False


def inverse_sqrt(X):
    """
    Compute the inverse square root of a positive definite matrix X using eigenvalue decomposition.
    
    Parameters:
    - X: (n, n) torch.Tensor, positive definite matrix.
    
    Returns:
    - X_inv_sqrt: (n, n) torch.Tensor, the inverse square root of X.
    """
    # Compute the eigenvalues and eigenvectors of the matrix
    eigvals, eigvecs = torch.linalg.eigh(X, UPLO='U')
    # Compute the inverse square root of the eigenvalues
    eigvals_inv_sqrt = 1. / torch.sqrt(eigvals)
    # Reconstruct the inverse square root of the matrix
    X_inv_sqrt = eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.t()

    return X_inv_sqrt
