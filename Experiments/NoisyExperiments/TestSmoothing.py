import os
import pickle
import numpy as np

from src.DART.DART import DART
from src.DART.DARTConfig import Config
from src.DART.Sinograms import PoissonNoise
from src.PhantomGenerators.Phantom4 import CircleWithGeoShapes
from src.PhantomGenerators.Phantom3 import BinaryGranularPhantom
from src.PhantomGenerators.Phantom12 import make_phantom2, make_phantom1

from numpy.typing import NDArray

# Experiment parameters
SMOOTHING_VALUES = [None, 0.5, 1, 2, 3]
SNR = [10, 20, 30]
PHANTOMS = [make_phantom1, make_phantom2, BinaryGranularPhantom, CircleWithGeoShapes]
N_ITERS = 10

PHANTOM_NAMES = {
    make_phantom1: "phantom_1",
    make_phantom2: "phantom_2",
    BinaryGranularPhantom: "phantom_3",
    CircleWithGeoShapes: "phantom_4",

}

config = Config()

def Reconstruct(phantom: NDArray, cfg: Config):
    """Container function for calling DART in the experiment"""

    _, results = DART(phantom=phantom, 
                      graylevels=np.array(cfg.gray_values),
                      p=cfg.p,
                      dart_iters=cfg.dart_iters,

                      init_arm_iters=cfg.init_arm_iters,
                      arm_iters=cfg.arm_iters,

                      angles=np.linspace(cfg.angle_range[0], cfg.angle_range[1], cfg.n_angles, endpoint=False),
                      detector_spacing=1,
                      n_detectors=512,

                      SNR=cfg.snr,
                      noise_func=PoissonNoise,

                      smoothing=cfg.sigma,

                      vol_data=0,
                      use_gpu=cfg.gpu)
    
    return results

def RunSmoothing(
                  dart_config: Config, 
                  smoothing_values: list[float] = SMOOTHING_VALUES, 
                  snr: list[int] = SNR, 
                  phantoms=PHANTOMS) -> None:

    """Sweeps over the smoothing values for all differnt phantoms and projections under noisy conditions"""

    for std in smoothing_values:
        # standard deviation for smoothing
        dart_config.sigma = std  

        for n in snr:
            # SNR for noisy projections
            dart_config.snr = n

            save_dir = os.path.join(dart_config.save_dir, "smoothing", f"{std}", f"snr_{n}")
            os.makedirs(save_dir, exist_ok=True)

            for phantom_gen in phantoms:
                # Phantoms
                file_name = PHANTOM_NAMES[phantom_gen]
                print(f"Running: Smoothing: {std} | SNR: {n} | Phantom: {PHANTOM_NAMES[phantom_gen]}")
                results = {}
                for i in range(N_ITERS):
                    path = os.path.join("TestPhantoms", str(PHANTOM_NAMES[phantom_gen]), str(i)+".npy")
                    phantom = np.load(path)
                    dart_config.gray_values = tuple(value for value in np.unique(phantom))
                    results[i] = Reconstruct(phantom, cfg=dart_config)

                save_path = os.path.join(save_dir, f"{file_name}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)
                
                print(f"Saved: {save_path}")


if __name__ == "__main__":

    dart_cfg = Config()
    dart_cfg.n_angles = 25
    dart_cfg.init_arm_iters = 50
    RunSmoothing( dart_config=dart_cfg, 
                  smoothing_values = SMOOTHING_VALUES, 
                  snr=SNR, 
                  phantoms=PHANTOMS)