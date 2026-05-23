import numpy as np
import skimage as sk
import os

from numpy.typing import NDArray
from Config import phantomConfig
from numpy.random import Generator
from src.PhantomGenerators.PhantomGenerators import Ellipse, Rectangle, Kite, RotateImage


def RandomGeoShape(cfg: phantomConfig, rng: Generator) -> NDArray:
    shape = rng.choice(["ellipse", "rectangle", "kite"])
    angle = rng.random(size=1) * 2 * np.pi
    img = np.zeros(cfg.img_shape)

    if shape == "ellipse":


    return np.zeros([])
