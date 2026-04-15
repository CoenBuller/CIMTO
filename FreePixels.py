import numpy as np

from EdgeDetector import EdgeDetection

def ChooseFreePixels(phantom: np.ndarray, p: float) -> np.ndarray:
    """
    This function will create a mask of the free pixels.

    ----------
    Parameters

    phantom: np.ndarray
        contains the phantom of which we want to decide the free pixels

    edge_mask: np.ndarray 
        masking array which contains the edge pixels

    p: float
        probability at which a pixel should be a free pixel
    """
    assert 0 <= p <= 1, "The free probability should be within the range [0, 1]"

    edge_mask = EdgeDetection(phantom=phantom)

    random_mask = np.random.choice(a=[0, 1], size=(phantom.shape), replace=True, p=[1-p, p]).astype(bool)
    free_pixel_mask = np.array((random_mask | edge_mask))

    return free_pixel_mask