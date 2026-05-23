import numpy as np
import os 
import time 

import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    time0 = time.time()
    edges = ChooseFreePixels(phantom=phantom, p=0.01)
    print(f"Free detection took {(time.time() - time0):.3f}s")

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom)
    ax[1].imshow(edges)
    ax[0].set_title("Originall phantom")
    ax[1].set_title("Free pixels")
    plt.show()
