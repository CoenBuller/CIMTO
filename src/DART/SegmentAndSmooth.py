import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def Smooth(phantom: NDArray, sigma: float, free_mask: NDArray) -> NDArray:
    """
    Apply Gaussian smoothing restricted to the free-pixel region.

    The full image is smoothed with a Gaussian kernel, but only the free pixels
    receive the smoothed values; fixed pixels are left untouched.  This avoids
    blurring the already-segmented boundary pixels while still suppressing
    noise in the uncertain regions before the next segmentation step.

    Parameters
    ----------
    phantom : NDArray
        Current (continuous) reconstruction.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.
    free_mask : NDArray
        Boolean mask; True where pixels are free (eligible for smoothing).

    Returns
    -------
    output : NDArray
        Reconstruction with smoothed values at free-pixel positions.
    """
    smoothed_full = gaussian_filter(input=phantom, sigma=sigma)

    output = phantom.copy()
    output[free_mask] = smoothed_full[free_mask]   # Overwrite only free pixels
    return output


def RoundTo(phantom: NDArray, graylevels: NDArray) -> NDArray:
    """
    Segment a continuous reconstruction by nearest-gray-level assignment.

    Each pixel value is independently mapped to the closest entry in
    `graylevels`.  This is the DART segmentation step that converts the
    continuous ARM output into a piecewise-constant image.

    Uses vectorised broadcasting to avoid explicit loops:
      - Broadcasting phantom[..., np.newaxis] against graylevels produces a
        (H, W, G) difference tensor.
      - argmin over the last axis gives the nearest gray-level index per pixel.

    Parameters
    ----------
    phantom : NDArray
        Continuous reconstruction of shape (H, W).
    graylevels : NDArray
        1-D array of known gray values (e.g. np.unique(true_phantom)).

    Returns
    -------
    NDArray
        Segmented image of shape (H, W) with values drawn from `graylevels`.
    """
    diffs = np.abs(phantom[..., np.newaxis] - graylevels)  # (H, W, G) absolute differences
    best_idx = np.argmin(diffs, axis=-1)                   # (H, W) index of nearest graylevel
    return graylevels[best_idx]

