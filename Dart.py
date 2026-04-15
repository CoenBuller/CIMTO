import numpy as np
import matplotlib.pyplot as plt
import os

from Sinograms import sinogram
from ReconstructionAlgorithms import SIRT
from EdgeDetector import EdgeDetection
from RoundTo import RoundTo
from FreePixels import ChooseFreePixels


def DART(phantom: np.ndarray, 
         graylevels: np.ndarray,
         p: float = 0,
         dart_iters: int = 200,

         iters: int = 200,
         angles: np.ndarray = np.linspace(0, np.pi, 180),
         detector_spacing: int = 1,
         n_detectors: int = 512,

         vol_data: np.ndarray | float = 0,
         use_gpu: bool = False,

) -> np.ndarray:
    
    projector_id, sino_id, sinogram_img, vol_geom, proj_geom = sinogram(phantom=phantom,
                                                                        n_detectors=n_detectors,
                                                                        angles=angles,
                                                                        detector_spacing=detector_spacing)

    reconstruction_id, reconstruction = SIRT(sinogram=sinogram_img,
                                             vol_geom=vol_geom,
                                             vol_data=vol_data,
                                             proj_geom=proj_geom,
                                             projector_id=projector_id,
                                             iters=iters,
                                             min_constraint=np.min(graylevels),
                                             max_constraint=np.max(graylevels),
                                             use_gpu=use_gpu)
    
    reconstruction = RoundTo(phantom=reconstruction, graylevels=graylevels)
    free_mask = ChooseFreePixels(reconstruction, p)

    for i in range(dart_iters - 1):

        reconstruction_id, reconstruction = SIRT(sinogram=sinogram_img,
                                                 mask=free_mask,
                                                 vol_geom=vol_geom,
                                                 vol_data=reconstruction,
                                                 proj_geom=proj_geom,
                                                 projector_id=projector_id,
                                                 iters=iters,
                                                 min_constraint=np.min(graylevels),
                                                 max_constraint=np.max(graylevels),
                                                 use_gpu=use_gpu)
        
        reconstruction = RoundTo(reconstruction, graylevels)
        free_mask = ChooseFreePixels(reconstruction, p)
        
    return reconstruction

if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    edge = EdgeDetection(phantom=phantom)
    free_pixels = ChooseFreePixels(phantom=phantom, p=0.1)

    # noise = np.random.normal(loc=3, scale=3, size=phantom.shape)
    # noisy_phantom = phantom + noise
    # graylevels = np.sort(np.unique(phantom))
    # rounded_phantom = RoundTo(phantom=noisy_phantom, graylevels=graylevels)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(edge)
    ax[0].set_title("Edge mask")
    ax[1].imshow(free_pixels)
    ax[1].set_title("Free pixels mask")
    ax[2].imshow(phantom)
    ax[2].set_title("Original phantom")
    plt.show()