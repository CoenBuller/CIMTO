import os
import pickle
import numpy as np

from src.DART.DART import DART
from src.DART.DARTConfig import Config
from src.DART.Sinograms import PoissonNoise
from src.PhantomGenerators.GeoPhantoms import CircleWithGeoShapes
from src.PhantomGenerators.GranularPhantoms import BinaryGranularPhantom
from src.PhantomGenerators.ConcavePhantoms import make_phantom2, make_phantom1
from src.PhantomGenerators.PhantomConfig import phantomConfig

from typing import Callable
from numpy.typing import NDArray
from numpy.random import Generator


INTERNAL_ARM_ITS = [1,3,5,10,20]
SNR = [10, 20, 30]
PHANTOMS = [make_phantom2, CircleWithGeoShapes]
N_ITERS = 10

PHANTOM_NAMES = {
    make_phantom2: "phantom_2",
    CircleWithGeoShapes: "phantom_4",
}

config = Config()

def Reconstruct(phantom: NDArray, cfg: Config):
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


def RunInternalARM(dart_config: Config, 
                  internal_arm_its: list[int] = INTERNAL_ARM_ITS, 
                  snr: list[int] = SNR, 
                  phantoms=PHANTOMS) -> None:

    for internal_arm in internal_arm_its:
        # Internal ARM iterations
        dart_config.arm_iters = internal_arm

        for n in snr:
            # SNR
            dart_config.snr = n

            save_dir = os.path.join(dart_config.save_dir, "InternalARM", f"{internal_arm}", f"snr_{n}")
            os.makedirs(save_dir, exist_ok=True)

            for phantom_gen in phantoms:
                # Phantoms
                file_name = PHANTOM_NAMES[phantom_gen]
                print(f"Running: Internal ARM Iters: {internal_arm} | SNR: {n} | Phantom: {PHANTOM_NAMES[phantom_gen]}")
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
    RunInternalARM(dart_config=dart_cfg, 
                  internal_arm_its = INTERNAL_ARM_ITS, 
                  snr= SNR, 
                  phantoms=PHANTOMS)