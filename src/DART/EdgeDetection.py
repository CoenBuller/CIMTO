import numpy as np

from scipy.ndimage import maximum_filter, minimum_filter

def EdgeDetection(phantom: np.ndarray) -> np.ndarray:
    """
    This function will detect the edges on a phantom and return a mask that contains the edges

    ----------
    Parameters:
    phantom: np.ndarray 
        contains the phantom image

    ----------
    Returns:
    mask: np.ndarray
        a mask that contains the edge
    """

    max_f = maximum_filter(phantom, size=3)
    min_f = minimum_filter(phantom, size=3)

    return max_f != min_f

