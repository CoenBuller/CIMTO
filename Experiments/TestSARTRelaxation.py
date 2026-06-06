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


RELAXATION_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0]
N_PROJECTIONS = [10, 25]
PHANTOMS = [make_phantom2, make_phantom1, BinaryGranularPhantom, CircleWithGeoShapes]
N_ITERS = 10

PHANTOM_NAMES = {
    make_phantom1: "phantom_1",
    make_phantom2: "phantom_2",
    BinaryGranularPhantom: "phantom_3",
    CircleWithGeoShapes: "phantom_4",
}


def Reconstruct(phantom: NDArray, cfg: Config):
    _, results = DART(
        phantom=phantom,
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
        relaxation=cfg.sart_relaxation,
        vol_data=0,
        use_gpu=cfg.gpu,
    )
    return results


def RunRelaxation(
    dart_config: Config,
    relaxation_values: list[float] = RELAXATION_VALUES,
    n_projections: list[int] = N_PROJECTIONS,
    phantoms=PHANTOMS,
) -> None:

    for lam in relaxation_values:
        dart_config.sart_relaxation = lam

        for n in n_projections:
            dart_config.n_angles = n

            save_dir = os.path.join(
                dart_config.save_dir, "sart_relaxation", f"{lam}", f"projections_{n}"
            )
            os.makedirs(save_dir, exist_ok=True)

            for phantom_gen in phantoms:
                file_name = PHANTOM_NAMES[phantom_gen]
                print(f"Running: Relaxation: {lam} | N projections: {n} | Phantom: {file_name}")
                results = {}
                for i in range(N_ITERS):
                    path = os.path.join("TestPhantoms", str(file_name), str(i) + ".npy")
                    phantom = np.load(path)
                    dart_config.gray_values = tuple(value for value in np.unique(phantom))
                    results[i] = Reconstruct(phantom, cfg=dart_config)

                save_path = os.path.join(save_dir, f"{file_name}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(results, f)

                print(f"Saved: {save_path}")


if __name__ == "__main__":
    dart_cfg = Config()
    RunRelaxation(
        dart_config=dart_cfg,
        relaxation_values=RELAXATION_VALUES,
        n_projections=N_PROJECTIONS,
        phantoms=PHANTOMS,
    )