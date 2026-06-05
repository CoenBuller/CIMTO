import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Any
from numpy.typing import NDArray
from src.DART.ReconstructionAlgorithms import SIRT
from src.DART.Sinograms import Sinogram
from scipy.ndimage import gaussian_filter



def Smooth(phantom: NDArray, sigma: float, free_mask: NDArray) -> NDArray:
    phantom = phantom * free_mask.astype(int)
    smooth = gaussian_filter(input=phantom, sigma=sigma)
    return smooth

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


if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "Porouse_with_different_density.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    projector_id, sino_id, sinogram_img, vol_geom, proj_geom = Sinogram(
        phantom=phantom,
        n_detectors=512,
        angles=np.linspace(0, np.pi, 180),
        detector_spacing=1
        )
    
    reconstruction= SIRT(
                            vol_data=0,
                            vol_geom=vol_geom,
                            projector_id=projector_id,
                            sino_id=sino_id,
                            min_constraint=0,
                            max_constraint=255
                            )
    
    np.save("Porouse_with_different_density", reconstruction)

    rounded_recon = RoundTo(reconstruction, graylevels=np.unique(phantom))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom)
    ax[1].imshow(rounded_recon)
    ax[0].set_title("Original phantom")
    ax[1].set_title("Rounded and reconstructed phantom")
    plt.show()

def ScaleTo(recon: NDArray, low_percentil: int=1, high_percentile: int=99, max_gray: int=255):
    low_val = np.percentile(recon, q=low_percentil)
    high_val = np.percentile(recon, q=high_percentile)
    print(low_val, high_val)

    scaled = max_gray * (recon - low_val)/(high_val - low_val)
    return scaled