import torch


SUPPORTED_BANDWIDTHS = ["scott", "silverman"]


def compute_bandwidth(X, bandwidth):
    """Generate a bandwidth matrix.

    scott factor: n**(-1./(d+4)). [1]
    silverman factor: (n * (d + 2) / 4.)**(-1. / (d + 4)). [2]

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    """
    if type(bandwidth) == float or isinstance(bandwidth, torch.Tensor) and bandwidth.dim() <= 1:
        bandwidth_ = bandwidth
    elif type(bandwidth) == str:
        cov = torch.cov(X.T)
        if bandwidth == "scott":
            bandwidth_ = cov * X.shape[0]**(-1./(X.shape[1]+4))
        else: # silverman
            bandwidth_ = cov * (X.shape[0]*((X.shape[1] + 2)/4.))**(-1./(X.shape[1]+4))
    
    return bandwidth_
