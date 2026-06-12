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
from src.DART.SART import SART
from src.DART.SegmentAndSmooth import RoundTo, Smooth
from src.DART.FreePixels import ChooseFreePixels
from src.DART.EdgeDetection import EdgeDetection



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
    
    # Will store results
    results = {"K_error" : [], "Abs_error": [], "time": 0}

    # Original sinogram
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
                          use_gpu=use_gpu,
                          proj_geom=proj_geom,
                          )

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

        # Bookkeeping 
        K_error = np.sum((reconstruction != phantom))
        abs_error = np.mean(abs(reconstruction - phantom))
        pbar.set_postfix(K=f"{K_error:.2f}",
                         abs_error=f"{abs_error}")
        pbar.update(1)  # account for the initial SIRT pass

        results["Abs_error"].append(abs_error)
        results["K_error"].append(K_error)

        # Actual loop
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
                                  use_gpu=use_gpu,
                                  proj_geom=proj_geom,
                                 )


            # Smooth the reconstruction
            if smoothing:
                reconstruction = Smooth(reconstruction, sigma=smoothing, free_mask=free_mask)
            continuous_reconstruction = reconstruction.copy()

            # Segment pixel values of reconstruction and determine new set of free pixels
            reconstruction = RoundTo(reconstruction, graylevels)
            free_mask = ChooseFreePixels(reconstruction, p)


            # More bookkeeping
            K_error = np.sum((reconstruction != phantom)) # Number of wrong pixels
            abs_error = np.mean(abs(reconstruction - phantom)) # Absolute error
            pbar.set_postfix(K=f"{K_error:.2f}",
                             abs_error=f"{abs_error}")
            pbar.update(1)

            results["Abs_error"].append(abs_error)
            results["K_error"].append(K_error)

    results["Time"] = time.time() - time0 
    astra.projector.delete(projector_id)
    return reconstruction, results

def ParseArgs():
    parser = argparse.ArgumentParser()

    # Phantom type
    parser.add_argument('-type', choices=['1', '2', '3', '4'], default='1')
    parser.add_argument('-instance', choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] , default='0')
    parser.add_argument('-verbal', type=bool, default=False)

    # DART parameters
    parser.add_argument('-p', type=float, default=0.85)
    parser.add_argument('-dart_iters', type=int, default=200)

    parser.add_argument('-arm_iters', type=int, default=3)
    parser.add_argument('-init_arm_iters', type=int, default=10)

    parser.add_argument('-lower_angle', type=float, default=0)
    parser.add_argument('-upper_angle', type=float, default=np.pi)
    parser.add_argument('-n_angles', type=int, default=25)
  

    parser.add_argument('-angle_ordering', type=str, default='random')
    parser.add_argument('-relaxation', type=float, default=1)
    parser.add_argument('-gpu', type=bool, default=False)

    parser.add_argument('-snr', type=int, default=None)
    parser.add_argument('-noise_func', type=str, default='poisson')

    parser.add_argument('-smoothing', type=float, default=1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """Can be used to run a single instance of the DART algorithm with custom parameter settings through the ArgsParser"""

    noise_funcs = {'poisson': PoissonNoise,}
    args = ParseArgs()
    phantom_path = os.path.join(f"TestPhantoms", f"phantom_{args.type}", f"{args.instance}.npy")
    phantom = np.load(phantom_path)

    reconstruction, results = DART(phantom=phantom,
                          graylevels=np.unique(phantom),
                          p=args.p,
                          dart_iters=args.dart_iters,

                          arm_iters=args.arm_iters,
                          init_arm_iters=args.init_arm_iters,

                          angles=np.linspace(args.lower_angle, args.upper_angle, args.n_angles, endpoint=True),
                          detector_spacing=1,
                          n_detectors=512,
                          angle_ordering=args.angle_ordering,
                          
                          relaxation=args.relaxation,
                          vol_data=0,
                          use_gpu=args.gpu,
                          
                          SNR=args.snr,
                          noise_func=noise_funcs[args.noise_func],
                          
                          smoothing=args.smoothing,
                          verbal=args.verbal)


    print('\n',"="*75)
    print(f"Final K: {np.sum(reconstruction != phantom)}")
    print("="*75)
    
    edge = EdgeDetection(reconstruction)
    not_in_edge = np.zeros_like(edge)
    not_in_edge[((phantom != reconstruction) not in edge)] = 1

    fig, ax = plt.subplots(1, 2)
    ax[0].axis("off")
    ax[1].axis("off")

    ax[0].imshow(reconstruction, cmap='viridis')
    ax[0].set_title("Reconstrction")
    ax[1].imshow(phantom, cmap='viridis')
    ax[1].set_title("Phantom")

    plt.tight_layout()
    plt.savefig("DART_reconstruction")
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(len(results["K_error"])), results["K_error"], 
            linewidth=3, color="#2E86AB", label='K-error')
    ax.set_xlabel("Iterations", fontsize=18)
    ax.set_ylabel("K-error", fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(loc='best', fontsize=18, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("convergence_plot", dpi=300, bbox_inches='tight') 

