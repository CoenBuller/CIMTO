import numpy as np
import scipy as sp

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

    kernel = np.ones((3, 3))
    output = sp.signal.convolve(in1=phantom, in2=kernel, mode='same') / 9
    output = np.round(output, decimals=3)
    mask = (output != phantom)
    return mask