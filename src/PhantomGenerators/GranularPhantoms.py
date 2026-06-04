import argparse

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from numpy.random import Generator
from PhantomConfig import phantomConfig

from skimage.draw import ellipse
from PhantomGenerators import PlotPhantom 
from scipy.ndimage import gaussian_filter


############################################# Granular Shapes #############################################
 
def BinaryGranularPhantom(cfg: phantomConfig, rng: Generator) -> NDArray:

    sigma = rng.uniform(1, 3)
    threshold = rng.uniform(0.01, 0.1)
    noise = rng.standard_normal(cfg.img_shape)
    smooth = gaussian_filter(noise, sigma=sigma)
    clippedSmooth = np.clip(smooth, 0, 1)

    # Make the image binary
    clippedSmooth[clippedSmooth > threshold] = 1
    clippedSmooth[clippedSmooth <= threshold] = 0

    # Define circular field
    radius = 240
    center = np.array(cfg.img_shape) // 2
    rr, cc = ellipse(center[0], center[1], radius, radius, shape=cfg.img_shape)

    # Use circular field to define a mask
    mask = np.ones(cfg.img_shape)
    mask[rr, cc] = 0

    # Set everything outside the circle equal to 0
    clippedSmooth[mask.astype(bool)] = 0

    clippedSmooth *= cfg.max_gray
    return clippedSmooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plot Geo Phantoms", description="Creating geometrical phantoms which will be used for the performance measuring of DART on different gray value levels")
    parser.add_argument('-seed', const=None)
    args = parser.parse_args()


    cfg = phantomConfig()
    rng = np.random.default_rng(seed=args.seed)
    img1 = BinaryGranularPhantom(cfg=cfg, rng=rng)
    img2 = BinaryGranularPhantom(cfg=cfg, rng=rng)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()