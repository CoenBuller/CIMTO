import numpy as np

from src.DART.Sinograms import PoissonNoise
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Any, Callable


@dataclass 
class Config:
    # Default DART parameters
    
    # Outer loop params
    gray_values: tuple[Any, ...] = (0., 72.5, 145., 222.5, 255.)
    p: float = 0.85 # Must be in [0, 1]
    dart_iters: int = 200 
    gpu: bool = False

    # Smoothing 
    sigma: float = 1
    
    # ARM parameters
    arm_iters: int = 3
    init_arm_iters: int = 10
    sart_relaxation: float = 1.0  # SART relaxation factor (lambda)

    # Sinogram params
    n_angles: int = 10
    angle_range: tuple[float, float] = (0, np.pi)
    angle_order: str = "random"  # "sequential" | "random" | "maximally_separated"

    # Noise 
    snr: int | None = None
    noise_pattern: Callable = PoissonNoise

    # RNG and results
    seed: int = 69
    save_dir: str = "Results/"