import numpy as np
import numba
from typing import Tuple, Union


class RandomSampleGenerator:
    """
    A fast random sample generator using Numba JIT compilation.
    Supports multiple probability distributions.
    """

    @staticmethod
    @numba.jit(nopython=True)
    def normal(mean: float, std: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a normal distribution.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from normal distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, size)

    @staticmethod
    @numba.jit(nopython=True)
    def uniform(low: float, high: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a uniform distribution.
        
        Args:
            low: Lower bound
            high: Upper bound
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from uniform distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, size)

    @staticmethod
    @numba.jit(nopython=True)
    def exponential(scale: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from an exponential distribution.
        
        Args:
            scale: Scale parameter (1/lambda)
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from exponential distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.exponential(scale, size)

    @staticmethod
    @numba.jit(nopython=True)
    def binomial(n: int, p: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a binomial distribution.
        
        Args:
            n: Number of trials
            p: Probability of success
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from binomial distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.binomial(n, p, size)

    @staticmethod
    @numba.jit(nopython=True)
    def poisson(lam: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a Poisson distribution.
        
        Args:
            lam: Rate parameter (lambda)
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from Poisson distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.poisson(lam, size)

    @staticmethod
    @numba.jit(nopython=True)
    def gamma(shape: float, scale: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a gamma distribution.
        
        Args:
            shape: Shape parameter (alpha)
            scale: Scale parameter (beta)
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from gamma distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.gamma(shape, scale, size)

    @staticmethod
    @numba.jit(nopython=True)
    def beta(a: float, b: float, size: int, seed: int = None) -> np.ndarray:
        """Generate samples from a beta distribution.
        
        Args:
            a: Alpha parameter
            b: Beta parameter
            size: Number of samples
            seed: Random seed for reproducibility
            
        Returns:
            Array of samples from beta distribution
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.beta(a, b, size)


# Example usage and testing
if __name__ == "__main__":
    gen = RandomSampleGenerator()
    
    # Test normal distribution
    normal_samples = gen.normal(0, 1, 1000000, seed=42)
    print(f"Normal distribution: mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}")
    
    # Test uniform distribution
    uniform_samples = gen.uniform(0, 10, 1000000, seed=42)
    print(f"Uniform distribution: mean={uniform_samples.mean():.4f}, std={uniform_samples.std():.4f}")
    
    # Test exponential distribution
    exp_samples = gen.exponential(2.0, 1000000, seed=42)
    print(f"Exponential distribution: mean={exp_samples.mean():.4f}, std={exp_samples.std():.4f}")
    
    # Test binomial distribution
    binom_samples = gen.binomial(10, 0.5, 1000000, seed=42)
    print(f"Binomial distribution: mean={binom_samples.mean():.4f}, std={binom_samples.std():.4f}")
    
    # Test Poisson distribution
    poisson_samples = gen.poisson(3.0, 1000000, seed=42)
    print(f"Poisson distribution: mean={poisson_samples.mean():.4f}, std={poisson_samples.std():.4f}")
    
    # Test gamma distribution
    gamma_samples = gen.gamma(2.0, 2.0, 1000000, seed=42)
    print(f"Gamma distribution: mean={gamma_samples.mean():.4f}, std={gamma_samples.std():.4f}")
    
    # Test beta distribution
    beta_samples = gen.beta(2.0, 5.0, 1000000, seed=42)
    print(f"Beta distribution: mean={beta_samples.mean():.4f}, std={beta_samples.std():.4f}")
