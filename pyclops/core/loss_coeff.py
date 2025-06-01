from typing import final, Union
from abc import ABC, abstractmethod
import math
import torch

######################################################################
# Loss coefficient implementations to be used in loss handling.
# These coefficients output tensors with values in [0, 1] that
# can be multiplied by loss values.
######################################################################

class LossCoeff(ABC): # this isn't so useful now, but was useful before we could condition the final structures.
    """
    Abstract base class to handle loss function coefficients for cyclic conditioning.
    
    All loss coefficients evaluate to values in the range [0, 1].
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the loss coefficient."""
        self._cached_area = None
        self._cached_num_samples = None
    
    @abstractmethod
    def _eval(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Evaluate the loss function coefficient at time t.
        
        Args:
            t: A float or torch.Tensor with values in [0, 1]
            
        Returns:
            A float or torch.Tensor with values in [0, 1]
        """
        pass

    @final
    def __call__(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Evaluate the loss function coefficient at time t.
        
        Args:
            t: A float or torch.Tensor with values in [0, 1]
            
        Returns:
            A float or torch.Tensor with values in [0, 1]
        """
        w_t = self._eval(t)
        self._assert_in_range(w_t)
        return w_t

    @final
    @staticmethod
    def _assert_in_range(w_t: Union[float, torch.Tensor]) -> None:
        """Ensure values are in the range [0, 1]."""
        if isinstance(w_t, float):
            assert 0 <= w_t <= 1, 'Loss coefficient must be between 0 and 1.'
        else:
            assert torch.all((w_t >= 0) & (w_t <= 1)), 'Loss coefficient must be between 0 and 1.'
    
    def _reset_cache(self) -> None:
        """Reset the cached area calculation."""
        self._cached_area = None
        self._cached_num_samples = None
            
    def area(self, num_samples: int = 1000, force_recalc: bool = False) -> float:
        """
        Calculate the area under the curve (integral from 0 to 1).
        
        By default, this uses numerical integration with the trapezoidal rule.
        Subclasses can override with exact solutions when available.
        
        Args:
            num_samples: Number of samples to use for numerical integration
            force_recalc: Whether or not to force an area recalculation
            
        Returns:
            The area under the curve
        """
        # Check if we have a cached result with the same number of samples
        if self._cached_area is not None and self._cached_num_samples == num_samples and not force_recalc:
            return self._cached_area
            
        # Create evenly spaced points for trapezoidal rule
        xs = [i / (num_samples - 1) for i in range(num_samples)]
        
        # Evaluate function at each point
        ys = [float(self(float(x))) for x in xs]
        
        # Apply trapezoidal rule: area = (width/2) * (y0 + 2*y1 + 2*y2 + ... + 2*y(n-1) + yn)
        width = 1.0 / (num_samples - 1)
        summed = ys[0] + ys[-1] + 2.0 * sum(ys[1:-1])
        result = (width / 2.0) * summed
        
        # Cache the result
        self._cached_area = result
        self._cached_num_samples = num_samples
        
        return result


class PseudoGaussian(LossCoeff):
    """
    A loss coefficient with a Gaussian shape but non-normalized area.
    
    Allows customization of the mean (mu), width (s), and height (coeff).
    """

    def __init__(self, mu: float, s: float, coeff: float = 1.0):
        """
        Initialize a pseudo-Gaussian loss coefficient.
        
        Args:
            mu: The temporal location of the maximum (between 0 and 1)
            s: The width of the distribution (must be positive)
            coeff: The scaling coefficient (between 0 and 1)
        """
        super().__init__()
        assert 0 <= mu <= 1, 'mu must be a float between 0 and 1, inclusive'
        assert s > 0, 's must be a positive float'
        assert 0 <= coeff <= 1, 'coeff must be a float between 0 and 1, inclusive'

        self._mu = mu
        self._s = s
        self._coeff = coeff
        self._exact_area_cache = None
    
    @property
    def mu(self) -> float:
        """Return the mean of the distribution."""
        return self._mu
    
    @mu.setter
    def mu(self, value: float) -> None:
        """Set the mean of the distribution."""
        assert 0 <= value <= 1, 'mu must be a float between 0 and 1, inclusive'
        self._mu = value
        self._reset_cache()
        self._exact_area_cache = None
    
    @property
    def s(self) -> float:
        """Return the width of the distribution."""
        return self._s
    
    @s.setter
    def s(self, value: float) -> None:
        """Set the width of the distribution."""
        assert value > 0, 's must be a positive float'
        self._s = value
        self._reset_cache()
        self._exact_area_cache = None
    
    @property
    def coeff(self) -> float:
        """Return the scaling coefficient."""
        return self._coeff
    
    @coeff.setter
    def coeff(self, value: float) -> None:
        """Set the scaling coefficient."""
        assert 0 <= value <= 1, 'coeff must be a float between 0 and 1, inclusive'
        self._coeff = value
        self._reset_cache()
        self._exact_area_cache = None
    
    def gaussian_cdf(self, x):
        """
        Gaussian CDF without using scipy.
        """
        return 0.5 * (1 + math.erf((x - self._mu) / (self._s * math.sqrt(2))))
    
    def area(self, num_samples: int = 1000, force_recalc: bool = False) -> float:
        """
        Calculate the area under the curve using the analytical formula.
        
        Returns:
            The area under the curve for the truncated Gaussian on [0, 1].
        """
        # If we already calculated the exact area, return it
        if self._exact_area_cache is not None and not force_recalc:
            return self._exact_area_cache
            
        # Direct calculation of the area between [0, 1] using the Gaussian CDF
        cdf_0 = self.gaussian_cdf(0)
        cdf_1 = self.gaussian_cdf(1)
        
        # Area calculation using the difference of CDFs
        self._exact_area_cache = self._coeff * (cdf_1 - cdf_0)
        return self._exact_area_cache
    
    def _eval(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Evaluate the pseudo-Gaussian at time t."""
        if isinstance(t, float):
            # Handle float input
            return self._coeff * math.exp(-0.5 * ((t - self._mu) / self._s)**2)
        else:
            # Handle tensor input
            return self._coeff * torch.exp(-0.5 * ((t - self._mu) / self._s)**2)


class MaxwellBoltzmann(LossCoeff):
    """
    A loss coefficient implementing the Maxwell-Boltzmann distribution.
    """
    
    def __init__(self, alpha: float):
        """
        Initialize a Maxwell-Boltzmann distribution.
        
        Args:
            alpha: The scale parameter (must be positive)
        """
        super().__init__()
        assert alpha > 0, 'alpha must be a positive float'
        self._alpha = alpha
        self._exact_area_cache = None
    
    @property
    def alpha(self) -> float:
        """Return the scale parameter."""
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the scale parameter."""
        assert value > 0, 'alpha must be a positive float'
        self._alpha = value
        self._reset_cache()
        self._exact_area_cache = None
    
    def area(self, num_samples: int = 1000, force_recalc: bool = False) -> float:
        """
        Calculate the area under the curve analytically.
        
        The Maxwell-Boltzmann distribution has a specific analytical solution
        for the integral from 0 to 1.
        
        Returns:
            The area under the curve from 0 to 1
        """
        # If we already calculated the exact area, return it
        if self._exact_area_cache is not None and not force_recalc:
            return self._exact_area_cache
            
        # Calculate the cumulative distribution function at x=1
        # F(x) = sqrt(2/π) * (α^3)^-1 * ∫[0→x] t^2 * e^(-t^2/(2α^2)) dt
        # This can be solved analytically
        
        # Substitute u = x/(sqrt(2)*α)
        u = 1.0 / (math.sqrt(2) * self._alpha)
        
        # The indefinite integral is:
        # F(x) = erf(u) - sqrt(2/π) * x * e^(-x^2/(2α^2))
        result = math.erf(u) - math.sqrt(2/math.pi) * (1.0 / u) * math.exp(-u**2)
        
        # Since we're integrating from 0 to 1, and F(0) = 0
        # we just need to return F(1)
        self._exact_area_cache = result
        return result
    
    def _eval(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Evaluate the Maxwell-Boltzmann distribution at time t."""
        if isinstance(t, float):
            # Handle float input
            scale_factor = math.sqrt(2.0 / math.pi)
            return scale_factor * (t**2 / self._alpha**3) * math.exp(-t**2 / (2 * self._alpha**2))
        else:
            # Handle tensor input
            scale_factor = torch.sqrt(torch.tensor(2.0 / math.pi))
            return scale_factor * (t**2 / self._alpha**3) * torch.exp(-t**2 / (2 * self._alpha**2))


class Constant(LossCoeff):
    """
    A constant loss coefficient that returns the same value for all times.
    """
    
    def __init__(self, const: float):
        """
        Initialize a constant loss coefficient.
        
        Args:
            const: The constant value (between 0 and 1)
        """
        super().__init__()
        assert 0 <= const <= 1, 'const must be a float between 0 and 1, inclusive'
        self._const = const
        self._exact_area_cache = None
    
    @property
    def const(self) -> float:
        """Return the constant value."""
        return self._const
    
    @const.setter
    def const(self, value: float) -> None:
        """Set the constant value."""
        assert 0 <= value <= 1, 'const must be a float between 0 and 1, inclusive'
        self._const = value
        self._reset_cache()
        self._exact_area_cache = None
    
    def area(self, num_samples: int = 1000, force_recalc: bool = False) -> float:
        """
        Calculate the area under the curve analytically.
        
        For a constant function over [0,1], the area is exactly the constant value.
        
        Returns:
            The area under the curve
        """
        # If we already calculated the exact area, return it
        if self._exact_area_cache is not None and not force_recalc:
            return self._exact_area_cache
            
        self._exact_area_cache = self._const
        return self._const
    
    def _eval(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Evaluate the constant function at time t."""
        if isinstance(t, float):
            return self._const
        else:
            return self._const * torch.ones_like(t)


class Pulse(LossCoeff):
    """
    A pulse (rectangular) loss coefficient that is non-zero only between times t1 and t2.
    """
    
    def __init__(self, t1: float, t2: float, const: float):
        """
        Initialize a pulse loss coefficient.
        
        Args:
            t1: The start time of the pulse (between 0 and 1)
            t2: The end time of the pulse (between 0 and 1)
            const: The pulse height (between 0 and 1)
        """
        super().__init__()
        assert 0.0 <= t1 < 1.0, 't1 must be between 0 and 1 (exclusive upper bound)'
        assert 0.0 < t2 <= 1.0, 't2 must be between 0 and 1 (exclusive lower bound)'
        assert t1 < t2, 't1 must be less than t2'
        assert 0.0 <= const <= 1.0, 'const must be between 0 and 1, inclusive'

        self._t1 = t1
        self._t2 = t2
        self._const = const
        self._exact_area_cache = None
    
    @property
    def t1(self) -> float:
        """Return the start time of the pulse."""
        return self._t1
    
    @t1.setter
    def t1(self, value: float) -> None:
        """Set the start time of the pulse."""
        assert 0.0 <= value < 1.0, 't1 must be between 0 and 1 (exclusive upper bound)'
        assert value < self._t2, 't1 must be less than t2'
        self._t1 = value
        self._reset_cache()
        self._exact_area_cache = None
    
    @property
    def t2(self) -> float:
        """Return the end time of the pulse."""
        return self._t2
    
    @t2.setter
    def t2(self, value: float) -> None:
        """Set the end time of the pulse."""
        assert 0.0 < value <= 1.0, 't2 must be between 0 and 1 (exclusive lower bound)'
        assert self._t1 < value, 't1 must be less than t2'
        self._t2 = value
        self._reset_cache()
        self._exact_area_cache = None
    
    @property
    def const(self) -> float:
        """Return the pulse height."""
        return self._const
    
    @const.setter
    def const(self, value: float) -> None:
        """Set the pulse height."""
        assert 0.0 <= value <= 1.0, 'const must be between 0 and 1, inclusive'
        self._const = value
        self._reset_cache()
        self._exact_area_cache = None
    
    def area(self, num_samples: int = 1000, force_recalc: bool = False) -> float:
        """
        Calculate the area under the curve analytically.
        
        For a pulse function, the area is (t2 - t1) * const.
        
        Returns:
            The area under the curve
        """
        # If we already calculated the exact area, return it
        if self._exact_area_cache is not None and not force_recalc:
            return self._exact_area_cache
            
        self._exact_area_cache = self._const * (self._t2 - self._t1)
        return self._exact_area_cache

    def _eval(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Evaluate the pulse function at time t."""
        if isinstance(t, float):
            # Handle float input
            return self._const if self._t1 <= t <= self._t2 else 0.0
        else:
            # Handle tensor input
            # Create condition masks
            lower_bound = (t >= self._t1)
            upper_bound = (t <= self._t2)
            in_range = lower_bound & upper_bound
            
            # Apply the pulse
            return torch.where(in_range, self._const, torch.zeros_like(t))