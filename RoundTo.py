import numpy as np

def RoundTo(phantom: np.ndarray, graylevels: np.ndarray) -> np.ndarray:
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

    phantom_copy = phantom.copy()
    graylevels_dict = {}
    for i, graylevel in enumerate(graylevels):
        graylevels_dict[i] = graylevel

    for i in range(phantom_copy.shape[0]):
        for j in range(phantom_copy.shape[1]):
            best_idx = np.argmin(np.abs(graylevels - phantom[i, j]))
            phantom_copy[i, j] = graylevels_dict[best_idx]

    return phantom_copy