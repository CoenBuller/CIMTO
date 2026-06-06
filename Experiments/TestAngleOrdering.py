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


ANGLE_ORDERINGS = ["sequential", "randomized", "maximally_separated"]
N_PROJECTIONS = [10, 25]
PHANTOMS = [make_phantom2, make_phantom1, BinaryGranularPhantom, CircleWithGeoShapes]
N_ITERS = 10

PHANTOM_NAMES = {
    make_phantom1: "phantom_1",
    make_phantom2: "phantom_2",
    BinaryGranularPhantom: "phantom_3",
    CircleWithGeoShapes: "phantom_4",
}


def GenerateAngles(cfg: Config) -> NDArray:
    """Generate projection angles according to the chosen ordering strategy."""
    start, stop = cfg.angle_range
    n = cfg.n_angles
    rng = np.random.default_rng(seed=cfg.seed)

    if cfg.angle_order == "sequential":
        return np.linspace(start, stop, n, endpoint=False)

    elif cfg.angle_order == "random":
        return np.sort(rng.uniform(start, stop, size=n))

    elif cfg.angle_order == "maximally_separated":
        # Golden-ratio sampling: each new angle maximally separates from previous
        golden = np.pi * (3.0 - np.sqrt(5.0))
        angles = (np.arange(n) * golden) % (stop - start) + start
        return np.sort(angles)

    else:
        raise ValueError(f"Unknown angle_order: {cfg.angle_order}")


def Reconstruct(phantom: NDArray, cfg: Config):
    _, results = DART(
        phantom=phantom,
        graylevels=np.array(cfg.gray_values),
        p=cfg.p,
        dart_iters=cfg.dart_iters,
        init_arm_iters=cfg.init_arm_iters,
        arm_iters=cfg.arm_iters,
        angle_ordering=cfg.angle_order,
        angles=np.linspace(cfg.angle_range[0], cfg.angle_range[1], cfg.n_angles, endpoint=False),
        detector_spacing=1,
        n_detectors=512,
        SNR=cfg.snr,
        noise_func=PoissonNoise,
        smoothing=cfg.sigma,
        vol_data=0,
        use_gpu=cfg.gpu,
    )
    return results


def RunAngleOrdering(
    dart_config: Config,
    angle_orderings: list[str] = ANGLE_ORDERINGS,
    n_projections: list[int] = N_PROJECTIONS,
    phantoms=PHANTOMS,
) -> None:

    for ordering in angle_orderings:
        dart_config.angle_order = ordering

        for n in n_projections:
            dart_config.n_angles = n

            save_dir = os.path.join(
                dart_config.save_dir, "angle_ordering", f"{ordering}", f"projections_{n}"
            )
            os.makedirs(save_dir, exist_ok=True)

            for phantom_gen in phantoms:
                file_name = PHANTOM_NAMES[phantom_gen]
                print(f"Running: Ordering: {ordering} | N projections: {n} | Phantom: {file_name}")
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
    RunAngleOrdering(
        dart_config=dart_cfg,
        angle_orderings=ANGLE_ORDERINGS,
        n_projections=N_PROJECTIONS,
        phantoms=PHANTOMS,
    )