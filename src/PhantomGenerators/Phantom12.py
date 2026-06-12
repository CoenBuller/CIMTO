import argparse
import os
import numpy as np

from numpy.typing import NDArray
from numpy.random import Generator

from src.PhantomGenerators.PhantomConfig import phantomConfig
from Plotting.PlotPhantom import PlotPhantom 


from skimage.draw import disk
from skimage.draw import polygon as sk_polygon
from skimage.morphology import dilation, erosion, disk
from scipy.ndimage import gaussian_filter1d


############################################# Concave Random Shapes #############################################

# parameters 
SIZE        = 512          # output image side length (pixels)
N_BLOBS     = 7            # number of random convex sub-shapes to union
MIN_RADIUS  = 30           # minimum blob radius  (pixels)
MAX_RADIUS  = 110          # maximum blob radius  (pixels)
N_VERTS     = 12           # polygon vertices per blob (more = rounder)
DILATE_R    = 20           # dilation radius to merge and smooth gaps
ERODE_R     = 8            # light erosion after dilation (optional cleanup)
 
# How much to erode to create interior grey shells
INNER_ERODE_1 = 40         # erode this much to get mid-grey boundary
INNER_ERODE_2 = 80         # erode this much more to get white interior

# Boundary perturbation (jagged edges)
N_PERTURB     = 80         # number of perturbation spots along the boundary
PERTURB_R_MIN = 4          # min radius of each perturbation disk (pixels)
PERTURB_R_MAX = 40         # max radius of each perturbation disk (pixels)
 
 
def random_convex_polygon(cx: int, cy: int, min_r: float, max_r: float, n_verts: int, rng: Generator,  r_noise: float | None = None):
    """Random convex polygon centred at (cx, cy)."""

    angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))

    # random radii with some angular wobble for irregularity
    radii  = rng.uniform(min_r, max_r, n_verts)
    rows = (cy + radii * np.sin(angles)).astype(float)
    cols = (cx + radii * np.cos(angles)).astype(float)

    return rows, cols


def random_nonconvex_polygon(cx: int, cy: int, min_r: float, max_r: float, n_verts: int, rng: Generator):
    """
    Star-shaped (non-convex) polygon centred at (cx, cy).
    Radii alternate between spike (large) and valley (small) values,
    forcing deep indentations without needing to sort angles.
    """
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    angles += rng.uniform(0, 2 * np.pi)   # random overall rotation

    spike_radii  = rng.uniform(min_r * 0.8, max_r,       n_verts)
    valley_radii = rng.uniform(min_r * 0.2, min_r * 0.7, n_verts)
    radii = np.where(np.arange(n_verts) % 2 == 0, spike_radii, valley_radii)

    rows = (cy + radii * np.sin(angles)).astype(float)
    cols = (cx + radii * np.cos(angles)).astype(float)
    return rows, cols


def perturb_boundary(mask: NDArray, n_spots: int, r_min: int, r_max: int, rng: Generator) -> NDArray:
    """
    Randomly add or remove small irregular blobs along the boundary of a
    binary mask to produce a jagged, irregular edge.
    Each blob is a small non-convex polygon (reusing random_nonconvex_polygon)
    so the stamps themselves have angular, uneven shapes rather than smooth circles.
    """
    eroded   = erosion(mask, footprint=disk(1))
    boundary = mask & ~eroded

    brows, bcols = np.where(boundary)
    if len(brows) == 0:
        return mask

    result = mask.copy()
    size   = mask.shape[0]

    for _ in range(n_spots):
        idx = rng.integers(len(brows))
        r, c = int(brows[idx]), int(bcols[idx])
        rad  = int(rng.integers(r_min, r_max + 1))

        # Build a small non-convex blob centred at (r, c)
        rows, cols = random_nonconvex_polygon(c, r, rad * 0.4, rad, 8, rng)
        rr, cc_px  = sk_polygon(rows, cols, shape=(size, size))

        if rng.random() < 0.5:
            result[rr, cc_px] = True    # add material (bump out)
        else:
            result[rr, cc_px] = False   # remove material (bite in)

    return result


def make_blob_mask(size, n_blobs, min_r, max_r, n_verts,
                   dilate_r, erode_r, rng,
                   jagged: bool = True,
                   n_perturb: int = N_PERTURB,
                   perturb_r_min: int = PERTURB_R_MIN,
                   perturb_r_max: int = PERTURB_R_MAX):

    mask = np.zeros((size, size), dtype=bool)

    margin = max_r + dilate_r + 10
    centres_r = rng.integers(margin, size - margin, size=n_blobs)
    centres_c = rng.integers(margin, size - margin, size=n_blobs)

    for cr, cc in zip(centres_r, centres_c):
        # Rasterise one blob into its own temporary mask
        rows, cols = random_convex_polygon(cc, cr, min_r, max_r, n_verts, rng)
        rr, cc_px  = sk_polygon(rows, cols, shape=(size, size))
        blob = np.zeros((size, size), dtype=bool)
        blob[rr, cc_px] = True

        # Perturb this blob's boundary before merging
        if jagged:
            blob = perturb_boundary(blob, n_perturb, perturb_r_min, perturb_r_max, rng)

        mask |= blob

    # Dilation merges overlapping/nearby blobs
    if dilate_r > 0:
        mask = dilation(mask, footprint=disk(dilate_r))

    # Erosion trims back the dilation overshoot
    if erode_r > 0:
        mask = erosion(mask, footprint=disk(erode_r))

    return mask
 
 
def make_phantom1(cfg: phantomConfig, n_blobs=N_BLOBS,
                  min_r=MIN_RADIUS, max_r=MAX_RADIUS,
                  n_verts=N_VERTS, dilate_r=DILATE_R,
                  erode_r=ERODE_R, rng=None,
                  jagged: bool = True,
                  n_perturb: int = N_PERTURB,
                  perturb_r_min: int = PERTURB_R_MIN,
                  perturb_r_max: int = PERTURB_R_MAX,
                  return_mask: bool = False):
    
    """Binary phantom (0 / 255). Pass jagged=True for a rougher boundary."""
    if rng is None:
        rng = np.random.default_rng(42)
    mask = make_blob_mask(cfg.img_shape[0], n_blobs, min_r, max_r,
                          n_verts, dilate_r, erode_r, rng,
                          jagged=jagged,
                          n_perturb=n_perturb,
                          perturb_r_min=perturb_r_min,
                          perturb_r_max=perturb_r_max)
    img = np.where(mask, 255, 0).astype(np.uint8)
    if return_mask: 
        return img, mask
    return img
 
 
def make_phantom2(cfg, rng, inner_erode_1=INNER_ERODE_1,
                  inner_erode_2=INNER_ERODE_2, smoothing: float = 1):
    """
    Three-grey-level phantom derived from a binary mask.
      - background  :   0
      - outer shell : 127
      - inner core  : 255
    The shells are produced by successive erosions of the binary mask.
    """

    img, mask = make_phantom1(cfg, n_blobs=N_BLOBS, min_r=MIN_RADIUS, max_r=MAX_RADIUS,
                            n_verts=N_VERTS, rng=rng, jagged=True, return_mask=True)
    
    shell1 = erosion(mask, footprint=disk(inner_erode_1))
    shell1 = gaussian_filter1d(shell1, sigma=smoothing)
    shell2 = erosion(mask, footprint=disk(inner_erode_2))
    shell2 = gaussian_filter1d(shell2, sigma=smoothing)
 
    img = np.zeros(mask.shape, dtype=np.uint8)
    img[mask]   = 127   # outer (mid-grey) region
    img[shell1] = 191   # type: ignore
    img[shell2] = 255   # type: ignore
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Plot Geo Phantoms", description="Creating geometrical phantoms which will be used for the performance measuring of DART on different gray value levels")
    parser.add_argument('-seed', const=None)
    parser.add_argument('-type', choices=['1', '2'])
    parser.add_argument('-jagged', action='store_true',
                        help="Use non-convex sub-shapes and boundary perturbation for a rougher edge")
    args = parser.parse_args()


    cfg = phantomConfig()
    rng = np.random.default_rng(seed=args.seed)

    if args.type == '1':
        img = make_phantom1(cfg, n_blobs=N_BLOBS, min_r=MIN_RADIUS, max_r=MAX_RADIUS,
                            n_verts=N_VERTS, rng=rng, jagged=args.jagged)
    elif args.type == '2':
        img = make_phantom2(cfg, rng, inner_erode_1=INNER_ERODE_1, inner_erode_2=INNER_ERODE_2)
    else:
        img = make_phantom1(cfg, n_blobs=N_BLOBS, min_r=MIN_RADIUS, max_r=MAX_RADIUS,
                            n_verts=N_VERTS, rng=rng, jagged=args.jagged)
    PlotPhantom(img) # type: ignore
    path = os.path.join("Test_phantoms", "phantom_"+args.type)
    np.savez(path, img)