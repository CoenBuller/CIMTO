import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.typing import NDArray
from src.DART.Sinograms import Sinogram
from scipy.ndimage import gaussian_filter



def Smooth(phantom: NDArray, sigma: float, free_mask: NDArray) -> NDArray:
    smoothed_full = gaussian_filter(input=phantom, sigma=sigma)

    output = phantom.copy()
    output[free_mask] = smoothed_full[free_mask]
    return output


def RoundTo(phantom: NDArray, graylevels: NDArray) -> NDArray:
    """
    Function rounds each pixel value to the nearest graylevel in a set of given graylevels

    ----------
    Parameters
    
    phantom: np.ndarray
        contains the phantom. Each value is a pixel-value
    
    graylevels: list 
        contains all the possible graylevels

    ----------
    Returns:
    np.ndarray:
        containing the pixels which are rounded to the nearest gray level. 
    """

    diffs = np.abs(phantom[..., np.newaxis] - graylevels)  # shape (H, W, G)
    best_idx = np.argmin(diffs, axis=-1)                   # shape (H, W)
    return graylevels[best_idx]


def ScaleTo(recon: NDArray, low_percentil: int=1, high_percentile: int=99, max_gray: int=255):
    low_val = np.percentile(recon, q=low_percentil)
    high_val = np.percentile(recon, q=high_percentile)
    print(low_val, high_val)

    scaled = max_gray * (recon - low_val)/(high_val - low_val)
    return scaled