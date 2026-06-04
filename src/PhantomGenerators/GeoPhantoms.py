import argparse

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from numpy.random import Generator
from PhantomConfig import phantomConfig

from skimage.draw import disk, random_shapes
from skimage.measure import label
from PhantomGenerators import PlotPhantom 
from scipy.ndimage import gaussian_filter


############################################# Circle With Geo Shapes #############################################
 
def CircleWithGeoShapes(cfg: phantomConfig, rng: Generator):
    gray_values = cfg.gray_values
    img = np.zeros(cfg.img_shape)

    w_disk_c, w_disk_r = (256, 256), 220
    rr, cc = disk(w_disk_c, w_disk_r, shape=img.shape)
    img[rr, cc] = cfg.max_gray

    b_disk_c, b_disk_r = (256, 256), 200
    rr, cc = disk(b_disk_c, b_disk_r, shape=img.shape)
    img[rr, cc] = cfg.min_gray

    shapes_img, labels = random_shapes(
                                        (256, 256), 
                                        max_shapes=15, 
                                        min_shapes=10, 
                                        min_size=30, 
                                        max_size=80, 
                                        intensity_range=(1, 254,),  # Fix 1: avoid 255 (background)
                                        rng=rng, 
                                        allow_overlap=False,
                                        num_channels=1,
                                    )

    shapes_mask = shapes_img[:, :, 0]
    labeled_shapes, num_features = label(shapes_mask, background=255, return_num=True)  # type: ignore
    center_area = np.zeros((256, 256))

    for shape_id in range(1, num_features + 1):
        random_color = rng.choice(gray_values)
        center_area[labeled_shapes == shape_id] = random_color

    img[128:512-128, 128:512-128] = center_area
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plot Geo Phantoms", description="Creating geometrical phantoms which will be used for the performance measuring of DART on different gray value levels")
    parser.add_argument('-seed', const=None)
    args = parser.parse_args()


    cfg = phantomConfig()
    rng = np.random.default_rng(seed=args.seed)
    img = CircleWithGeoShapes(cfg=cfg, rng=rng)

    PlotPhantom(img)