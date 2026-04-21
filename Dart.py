import numpy as np
import matplotlib.pyplot as plt
import os
import astra
from tqdm import tqdm
from Sinograms import Sinogram, ResidualSinogram
from ReconstructionAlgorithms import SIRT
from RoundTo import RoundTo
from FreePixels import ChooseFreePixels


def DART(phantom: np.ndarray,
         graylevels: np.ndarray,
         p: float = 0,
         dart_iters: int = 50,

         init_sirt_iters: int = 50,
         sirt_iters: int = 10,
         angles: np.ndarray = np.linspace(0, np.pi, 180),
         detector_spacing: int = 1,
         n_detectors: int = 512,
         intensity_scale: int | None= None,

         vol_data: np.ndarray | float = 0,
         use_gpu: bool = False,
         stagnated_iteraions: int = 3,

        ) -> np.ndarray:

    projector_id, sino_id, sinogram_img, vol_geom, proj_geom = Sinogram(
                                                                        phantom=phantom,
                                                                        n_detectors=n_detectors,
                                                                        angles=angles,
                                                                        detector_spacing=detector_spacing,
                                                                        intensity_scale=intensity_scale,
                                                                        )

    # Initial reconstruction
    reconstruction = SIRT(
                          sino_id=sino_id,
                          vol_geom=vol_geom, 
                          vol_data=vol_data, 
                          projector_id=projector_id,
                          iters=init_sirt_iters,
                          min_constraint=np.min(graylevels),
                          max_constraint=np.max(graylevels),
                          use_gpu=use_gpu
                          )

    reconstruction = RoundTo(phantom=reconstruction, graylevels=graylevels)
    free_mask = ChooseFreePixels(reconstruction, p)

    print('\n', "="*75)
    print("Initial reconstruction has been made. Will now continue with the DART loop")
    print("="*75, '\n')
    
    with tqdm(total=dart_iters, desc="DART", unit="iter") as pbar:

        K_error = np.sum((reconstruction != phantom))
        abs_error = np.mean(abs(reconstruction - phantom))
        pbar.set_postfix(K=f"{K_error:.2f}",
                         abs_error=f"{abs_error}")
        pbar.update(1)  # account for the initial SIRT pass

        stagnated = 0
        K_error_prev = K_error
        for i in range(dart_iters - 1):

            # Calculate residual sinogram b_res = b_0 - A(x_fixed)
            residual_sino_id = ResidualSinogram(reconstruction=reconstruction,
                                                free_mask=free_mask,
                                                sinogram_img=sinogram_img,
                                                projector_id=projector_id,
                                                vol_geom=vol_geom, 
                                                projector_geom=proj_geom)
            
            # Solve b_res = A(x_free)
            reconstruction = SIRT(
                sino_id=residual_sino_id,
                mask=free_mask,
                vol_geom=vol_geom,
                vol_data=reconstruction,
                projector_id=projector_id,
                iters=sirt_iters,
                min_constraint=np.min(graylevels),
                max_constraint=np.max(graylevels),
                use_gpu=use_gpu
            )
            
            # Segment pixel values of reconstruction and determine new set of free pixels
            reconstruction = RoundTo(reconstruction, graylevels)
            free_mask = ChooseFreePixels(reconstruction, p)

            # Some metrics
            K_error = np.sum((reconstruction != phantom)) # Number of wrong pixels
            abs_error = np.mean(abs(reconstruction - phantom))
            pbar.set_postfix(K=f"{K_error:.2f}",
                             abs_error=f"{abs_error}")
            pbar.update(1)

            if K_error == K_error_prev:
                stagnated += 1
                if stagnated == stagnated_iteraions: # If stagnated for "stagnated_iterations" stop DART loop
                    break
            
            K_error_prev = K_error

    astra.projector.delete(projector_id)
    return reconstruction

if __name__ == "__main__":
    phantom_path = os.path.join("Test_phantoms", "multiple_shapes_and_graylevels.npz")
    phantom_arrays = np.load(phantom_path)
    lst = phantom_arrays.files
    item = lst[0]
    phantom = phantom_arrays[item]

    reconstruction = DART(phantom=phantom,
                          graylevels=np.unique(phantom),
                          p=0.1,
                          dart_iters=20,

                          sirt_iters=20,
                          init_sirt_iters=200,

                          angles=np.linspace(0, np.pi, 180),
                          detector_spacing=1,
                          n_detectors=512,

                          vol_data=0,
                          use_gpu=False,
                          stagnated_iteraions=5,)


    print('\n',"="*75)
    print(f"Final K: {np.sum(reconstruction != phantom)}")
    print("="*75)

    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phantom)
    ax[0].set_title("Original phantom")
    ax[1].imshow(reconstruction)
    ax[1].set_title("Reconstructed phantom")
    plt.show()


