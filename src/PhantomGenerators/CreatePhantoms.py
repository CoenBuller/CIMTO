from src.PhantomGenerators.Phantom4 import CircleWithGeoShapes
from src.PhantomGenerators.Phantom3 import BinaryGranularPhantom
from src.PhantomGenerators.Phantom12 import make_phantom2, make_phantom1
from src.PhantomGenerators.PhantomConfig import phantomConfig

from numpy.random import Generator
import numpy as np
import os


PHANTOMS = [make_phantom2, make_phantom1, BinaryGranularPhantom, CircleWithGeoShapes]
N_ITERS = 10

PHANTOM_NAMES = {
    make_phantom1: "phantom_1",
    make_phantom2: "phantom_2",
    BinaryGranularPhantom: "phantom_3",
    CircleWithGeoShapes: "phantom_4",
}

def CreatePhantoms(phantom_cfg: phantomConfig, rng: Generator):

    for phantom_gen in PHANTOMS:
        save_dir = os.path.join(phantom_cfg.save_dir, PHANTOM_NAMES[phantom_gen])
        os.makedirs(save_dir, exist_ok=True)

        for iter in range(N_ITERS):

            phantom = phantom_gen(cfg=phantom_cfg, rng=rng)
            save_path = os.path.join(save_dir, f"{iter}")
            np.save(save_path, arr=phantom)

if __name__ == "__main__":
    cfg = phantomConfig()
    rng = np.random.default_rng(seed=cfg.seed)
    CreatePhantoms(phantom_cfg=cfg, rng=rng)