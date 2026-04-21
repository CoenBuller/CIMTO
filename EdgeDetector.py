import numpy as np
import os

from scipy.ndimage import maximum_filter, minimum_filter

import time
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    time0 = time.time()
    edges = EdgeDetection(phantom=phantom)
    print(f" Edge detection took {(time.time() - time0):.3f}s")

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom)
    ax[1].imshow(edges)
    ax[0].set_title("Originall phantom")
    ax[1].set_title("Edges")
    plt.show()
