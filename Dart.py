import numpy as np
import matplotlib.pyplot as plt
import os
import astra
from tqdm import tqdm
from Sinograms import sinogram
from ReconstructionAlgorithms import SIRT
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

    projector_id, sino_id, sinogram_img, vol_geom, proj_geom = sinogram(
        phantom=phantom,
        n_detectors=n_detectors,
        angles=angles,
        detector_spacing=detector_spacing
    )
    astra.data2d.delete(sino_id)


    reconstruction = SIRT(
        sinogram=sinogram_img,
        vol_geom=vol_geom, 
        vol_data=vol_data, 
        proj_geom=proj_geom, 
        projector_id=projector_id,
        iters=iters,
        min_constraint=np.min(graylevels),
        max_constraint=np.max(graylevels),
        use_gpu=use_gpu
    )

    reconstruction = RoundTo(phantom=reconstruction, graylevels=graylevels) # type: ignore
    free_mask = ChooseFreePixels(reconstruction, p)

    
    with tqdm(total=dart_iters, desc="DART", unit="iter") as pbar:

        K_error = np.sum((reconstruction != phantom))
        pbar.set_postfix(abs_error=f"{K_error:.2f}")
        pbar.update(1)  # account for the initial SIRT pass

        for i in range(dart_iters - 1):

            reconstruction = SIRT(
                sinogram=sinogram_img,
                mask=free_mask,
                vol_geom=vol_geom,
                vol_data=reconstruction,
                proj_geom=proj_geom,
                projector_id=projector_id,
                iters=iters,
                min_constraint=np.min(graylevels),
                max_constraint=np.max(graylevels),
                use_gpu=use_gpu
            )

            reconstruction = RoundTo(reconstruction, graylevels)
            free_mask = ChooseFreePixels(reconstruction, p)

            K_error = np.sum((reconstruction != phantom))
            pbar.set_postfix(abs_error=f"{K_error:.2f}")
            pbar.update(1)

    return reconstruction

if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    reconstruction = DART(phantom=phantom,
                          graylevels=np.unique(phantom),
                          p=0.00,
                          dart_iters=0,
                          iters=100,
                          angles=np.linspace(0, np.pi, 180),
                          detector_spacing=1,
                          n_detectors=512,
                          vol_data=0,
                          use_gpu=False)


    print("="*75)
    print(f"Final MAE: {np.sum(reconstruction != phantom)}")
    print("="*75)

    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom)
    ax[0].set_title("Original phantom")
    ax[1].imshow(reconstruction)
    ax[1].set_title("Reconstructed phantom")
    plt.show()


