import numpy as np
import matplotlib.pyplot as plt
import os

from Sinograms import sinogram
from EdgeDetector import EdgeDetection


def DART(phantom: np.ndarray, graylevels: list) -> np.ndarray:


    edge = EdgeDetection(phantom=phantom)
    plt.imshow(edge)

    return np.array([])

if __name__ == "__main__":
    phantom_path = os.path.join("CIMTO", "Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    edge = EdgeDetection(phantom=phantom)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(edge)
    ax[1].imshow(phantom)
    plt.show()