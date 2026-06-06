import numpy as np
import matplotlib.pyplot as plt
import os
import astra
import time
import argparse

from numpy.typing import NDArray
from typing import Callable
from tqdm import tqdm
from src.DART.Sinograms import Sinogram, ResidualSinogram, PoissonNoise
from src.DART.ReconstructionAlgorithms import SART
from src.DART.RoundTo import RoundTo, Smooth
from src.DART.FreePixels import ChooseFreePixels
from src.DART.EdgeDetector import EdgeDetection



def DART(phantom: NDArray,
         graylevels: NDArray,
         p: float = 1,
         dart_iters: int = 50,

         init_arm_iters: int = 50,
         arm_iters: int = 10,
         relaxation: float = 1.0,

         angle_ordering: str = "random",
         angles: NDArray = np.linspace(0, np.pi, 180),
         detector_spacing: int = 1,
         n_detectors: int = 512,

         SNR: int | None= None,
         noise_func: Callable = PoissonNoise,

         smoothing: float | None = 1,

         vol_data: NDArray | float = 0,
         use_gpu: bool = False,
         verbal: bool = False

        ) -> tuple[NDArray, dict[str, list[float]]]:
    

    results = {"K_error" : [], "Abs_error": [], "time": 0}

    projector_id, sino_id, sinogram_img, vol_geom, proj_geom = Sinogram(
                                                                        phantom=phantom,
                                                                        n_detectors=n_detectors,
                                                                        angles=angles,
                                                                        detector_spacing=detector_spacing,
                                                                        SNR=SNR,
                                                                        noise_func=noise_func
                                                                        )

    # Initial reconstruction
    time0 = time.time()
    reconstruction = SART(
                          sino_id=sino_id,
                          vol_geom=vol_geom, 
                          vol_data=vol_data, 
                          projector_id=projector_id,
                          iters=init_arm_iters,
                          relaxation=relaxation,
                          projection_order=angle_ordering,
                          min_constraint=np.min(graylevels),
                          max_constraint=np.max(graylevels),
                          use_gpu=use_gpu
                          )
    
    # Smooth and segmentate
    if smoothing:
        temp_rounded = RoundTo(phantom=reconstruction, graylevels=graylevels)
        edge_mask = EdgeDetection(temp_rounded)
        reconstruction = Smooth(reconstruction, sigma=smoothing, free_mask=np.ones_like(edge_mask))

    continuous_reconstruction = reconstruction.copy()          
    reconstruction = RoundTo(phantom=reconstruction, graylevels=graylevels)
    free_mask = ChooseFreePixels(reconstruction, p)

    if verbal:
        print('\n')
        print("="*75)
        print("Initial reconstruction has been made. Will now continue with the DART loop")
        print(f"Initial reconstruction took {(time.time() - time0):.3f}s, for {init_arm_iters} iterations")
        print("="*75, '\n')
        
    with tqdm(total=dart_iters, desc="DART", unit="iter") as pbar:

        K_error = np.sum((reconstruction != phantom))
        abs_error = np.mean(abs(reconstruction - phantom))
        pbar.set_postfix(K=f"{K_error:.2f}",
                         abs_error=f"{abs_error}")
        pbar.update(1)  # account for the initial SIRT pass

        results["Abs_error"].append(abs_error)
        results["K_error"].append(K_error)
        for i in range(dart_iters - 1):

            # Calculate residual sinogram b_res = b_0 - A(x_fixed)
            residual_sino_id = ResidualSinogram(reconstruction=reconstruction,
                                                free_mask=free_mask,
                                                sinogram_img=sinogram_img,
                                                projector_id=projector_id,
                                                vol_geom=vol_geom, 
                                                projector_geom=proj_geom)
            

            vol_init = reconstruction.copy()                          
            vol_init[free_mask] = continuous_reconstruction[free_mask]
            
            # Solve b_res = A(x_free)
            reconstruction = SART(
                                  sino_id=residual_sino_id,
                                  mask=free_mask,
                                  vol_geom=vol_geom,
                                  vol_data=vol_init,
                                  projector_id=projector_id,
                                  iters=arm_iters,
                                  projection_order=angle_ordering,
                                  relaxation=relaxation,
                                  min_constraint=np.min(graylevels),
                                  max_constraint=np.max(graylevels),
                                  use_gpu=use_gpu
                                 )


            # Smooth the reconstruction
            if smoothing:
                temp_rounded = RoundTo(reconstruction, graylevels)
                edge_mask = EdgeDetection(temp_rounded)
                reconstruction = Smooth(reconstruction, sigma=smoothing, free_mask=np.ones_like(edge_mask))

            continuous_reconstruction = reconstruction.copy()
            # Segment pixel values of reconstruction and determine new set of free pixels
            reconstruction = RoundTo(reconstruction, graylevels)
            free_mask = ChooseFreePixels(reconstruction, p)


            # Some metrics
            K_error = np.sum((reconstruction != phantom)) # Number of wrong pixels
            abs_error = np.mean(abs(reconstruction - phantom))
            pbar.set_postfix(K=f"{K_error:.2f}",
                             abs_error=f"{abs_error}")
            pbar.update(1)

            results["Abs_error"].append(abs_error)
            results["K_error"].append(K_error)

    results["Time"] = time.time() - time0 
    astra.projector.delete(projector_id)
    return reconstruction, results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-type', choices=['1', '2', '3', '4'], default='1')
    parser.add_argument('-instance', choices=['1', '2', '3', '4', '5', '6', '7', '8', '9'] , default='0')
    args = parser.parse_args()
    phantom_path = os.path.join(f"TestPhantoms", f"phantom_{args.type}", f"{args.instance}.npy")
    phantom = np.load(phantom_path)

    reconstruction, _ = DART(phantom=phantom,
                          graylevels=np.unique(phantom),
                          p=0.85,
                          dart_iters=200,

                          arm_iters=3,
                          init_arm_iters=10,

                          angles=np.linspace(0, np.pi, 10, endpoint=True),
                          detector_spacing=1,
                          n_detectors=512,
                          angle_ordering="sequential",
                    
                          vol_data=0,
                          use_gpu=False,
                          
                          SNR=None,
                          noise_func=PoissonNoise,
                          
                          smoothing=0.5)


    print('\n',"="*75)
    print(f"Final K: {np.sum(reconstruction != phantom)}")
    print("="*75)

    edge = EdgeDetection(reconstruction)
    not_in_edge = np.zeros_like(edge)
    not_in_edge[((phantom != reconstruction) not in edge)] = 1
    fig, ax = plt.subplots(2, 3)
    ax[0,0].imshow(phantom)
    ax[0,0].set_title("Original phantom")
    ax[0,1].imshow(reconstruction)
    ax[0,1].set_title("Reconstructed phantom")
    ax[1,0].imshow(phantom != reconstruction)
    ax[1,0].set_title("Difference phantom")
    ax[1,1].imshow(edge)
    ax[1,1].set_title("Edges")
    ax[1,2].imshow(not_in_edge)
    ax[1,2].set_title(f"All the pixel values that are wrong, that are not part of the edge. N={np.sum(not_in_edge)}")
    plt.savefig("DART_reconstruction")
    plt.show()


