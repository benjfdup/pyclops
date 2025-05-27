from abc import ABC, abstractmethod
import math

import torch
from scipy.special import gamma

from .utils import check_if_mat, inverse_sqrt


SUPPORTED_KERNELS = [
    "gaussian",
    "epanechnikov",
    "exponential",
    "tophat-approx",
    "cauchy",
]


class Kernel(ABC):
    def __init__(self):
        self._bandwidth = None
        self._norm_constant = None
        self.dim = None

    @property
    def bandwidth(self):
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth
        # compute H^(-1/2)
        if check_if_mat(bandwidth):
            self.inv_bandwidth = inverse_sqrt(bandwidth)
        else:  # Scalar case
            self.inv_bandwidth = self.bandwidth**(-0.5)
    
    @property
    def norm_constant(self):
        if self._norm_constant is None:
            assert self.dim is not None, "Dimension not set."
            self._norm_constant = self._compute_norm_constant(self.dim)
        return self._norm_constant

    @abstractmethod
    def _compute_norm_constant(self, dim):
        pass

    @abstractmethod
    def __call__(self, x):
        assert self.bandwidth is not None, "Bandwidth not set."


class GaussianKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        self.dim = x.shape[-1]
        c = self.norm_constant
        u = kernel_input(self.inv_bandwidth, x)
        return c*torch.exp(-u/2)

    def _compute_norm_constant(self, dim):
        # normalizing constant for the Gaussian kernel
        return 1/(2*math.pi)**(dim/2)


class TopHatKernel(Kernel):
    """Differentiable approximation of the top-hat kernel 
    via a generalized Gaussian."""
    def __init__(self, beta=8):
        super().__init__()
        assert type(beta) == int, "beta must be an integer."
        self.beta = beta

    def __call__(self, x):
        super().__call__(x)
        self.dim = x.shape[-1]
        c = self.norm_constant
        u = kernel_input(self.inv_bandwidth, x)
        return c*torch.exp(-(u**self.beta)/2)

    def _compute_norm_constant(self, dim):
        # normalizing constant for the Gaussian kernel
        # reference: https://arxiv.org/pdf/1302.6498
        return (self.beta*gamma(dim/2))/(math.pi**(dim/2) * \
                                         gamma(dim/(2*self.beta)) * 2**(dim/(2*self.beta)))


class EpanechnikovKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        self.dim = x.shape[-1]
        c = self.norm_constant
        u = kernel_input(self.inv_bandwidth, x)
        return torch.where(u > 1, 0, c * (1 - u))
    
    def _compute_norm_constant(self, dim):
        # normalizing constant for the Epanechnikov
        return ((dim + 2)*gamma(dim/2 + 1))/(2*math.pi**(dim/2))


class ExponentialKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        self.dim = x.shape[-1]
        c = self.norm_constant
        u = kernel_input(self.inv_bandwidth, x, exp=1)
        return c*torch.exp(-u)
    
    def _compute_norm_constant(self, dim):
        # normalizing constant for the exponential kernel
        return 1/(2**dim)
    

class CauchyKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        self.dim = x.shape[-1]
        c = self.norm_constant
        u = kernel_input(self.inv_bandwidth, x, exp=2)
        return c / (1 + u)**((self.dim + 1)/2)

    def _compute_norm_constant(self, dim):
        # Normalizing constant for the multivariate Cauchy distribution
        return gamma((dim + 1) / 2) / (math.pi**(dim / 2) * gamma(0.5))
    

def kernel_input(inv_bandwidth, x, exp=2):
    """Compute the input to the kernel function."""
    if exp >= 2:
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth)**exp).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth)**exp).sum(-1)
    else: # absolute value
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth).abs()).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth).abs()).sum(-1)
    