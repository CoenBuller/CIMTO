import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def segment_to_grey_levels(image: np.ndarray, grey_levels: np.ndarray) -> np.ndarray:
    """Assign each pixel to the nearest grey level by simple thresholding."""
    diffs = np.abs(image[:, :, np.newaxis] - grey_levels[np.newaxis, np.newaxis, :])
    return grey_levels[np.argmin(diffs, axis=2)]


def gaussian_value_smooth(
    reconstruction: np.ndarray,
    scale: int,
    sigma: float,
) -> np.ndarray:
    """
    Smooth the reconstruction using a Gaussian kernel centred on each
    pixel's own value. Each neighbour is weighted by:

        w(n, n') = exp( -(recon[n'] - recon[n])^2 / (2 * sigma^2) )

    so neighbours with similar values to the centre pixel contribute
    strongly, while neighbours across a boundary (very different value)
    contribute almost nothing. This means the smoothing respects edges
    rather than blurring across them, unlike a uniform filter.

    sigma should be set relative to the spacing between grey levels —
    a reasonable default is half the minimum inter-level spacing.

    Parameters
    ----------
    reconstruction : np.ndarray, shape (H, W)
    scale : int (odd)
        Neighbourhood size. scale=1 returns reconstruction unchanged.
    sigma : float
        Standard deviation of the value-space Gaussian. Controls how
        aggressively cross-boundary neighbours are down-weighted.

    Returns
    -------
    smoothed : np.ndarray, shape (H, W)
    """
    if scale == 1:
        return reconstruction.astype(float)

    pad = scale // 2
    padded = np.pad(reconstruction.astype(float), pad, mode="reflect")

    # windows shape: (H, W, scale, scale)
    windows = sliding_window_view(padded, (scale, scale))

    # Centre pixel value broadcast to match window shape
    centre = reconstruction[:, :, np.newaxis, np.newaxis]

    # Gaussian weights based on value distance from centre pixel
    weights = np.exp(-((windows - centre) ** 2) / (2.0 * sigma ** 2))

    smoothed = (weights * windows).sum(axis=(-2, -1)) / weights.sum(axis=(-2, -1))
    return smoothed


def _count_consistent_neighbours(segmented: np.ndarray, scale: int) -> np.ndarray:
    """
    For each pixel, count how many pixels in its (scale x scale) neighbourhood
    share the same segmentation label. Vectorised over grey levels.
    """
    q = scale * scale
    pad = scale // 2
    counts = np.zeros(segmented.shape, dtype=float)

    for gl in np.unique(segmented):
        mask = (segmented == gl).astype(float)
        padded = np.pad(mask, pad, mode="constant")
        windows = sliding_window_view(padded, (scale, scale))  # (H, W, scale, scale)
        neighbour_sum = windows.sum(axis=(-2, -1))
        counts += np.where(segmented == gl, neighbour_sum, 0)

    return counts


def lidar_multiscale_segment(
    reconstruction: np.ndarray,
    grey_levels: list,
    scales: list = [1, 3, 5, 7],
    sigma: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive multiscale segmentation with value-space Gaussian smoothing.

    For each pixel, iterates from coarsest to finest scale and selects
    the finest scale at which the pixel has at least sqrt(q_L) consistent
    neighbours (q_L = coarsest neighbourhood size). The smoothed value
    at that scale — computed with the Gaussian value kernel — is used
    for thresholding rather than the raw reconstruction value.

    The Gaussian kernel means smoothing does not blur across boundaries,
    so boundary pixels naturally get assigned to a coarser scale while
    deep interior pixels stay at the finest scale.

    Parameters
    ----------
    reconstruction : np.ndarray, shape (H, W)
    grey_levels : list of float
    scales : list of int (odd, finest first)
    sigma : float or None
        Value-space Gaussian sigma. Defaults to half the minimum spacing
        between grey levels — large enough to pool within a region,
        small enough to suppress cross-boundary contribution.

    Returns
    -------
    segmented : np.ndarray, shape (H, W)
    scale_map : np.ndarray of int, shape (H, W)
        Which scale was selected per pixel (for visualisation).
    """
    grey_levels = np.asarray(grey_levels, dtype=float)
    scales = sorted(scales)

    if sigma is None:
        spacings = np.diff(np.sort(grey_levels))
        sigma = float(spacings.min()) / 2.0

    q_L = scales[-1] ** 2
    consistency_threshold = np.sqrt(q_L)

    H, W = reconstruction.shape

    # Build smoothed reconstructions and segmentations at every scale
    smoothed_at = {}
    segmented_at = {}
    consistent_at = {}

    for scale in scales:
        s = gaussian_value_smooth(reconstruction, scale, sigma)
        seg = segment_to_grey_levels(s, grey_levels)
        smoothed_at[scale] = s
        segmented_at[scale] = seg
        consistent_at[scale] = _count_consistent_neighbours(seg, scale)

    # Start with coarsest scale as fallback for every pixel
    selected_smoothed = smoothed_at[scales[-1]].copy()
    scale_map = np.full((H, W), scales[-1], dtype=int)

    # Overwrite with finer scales where consistency criterion is met
    for scale in reversed(scales):
        consistent = consistent_at[scale] >= consistency_threshold
        selected_smoothed[consistent] = smoothed_at[scale][consistent]
        scale_map[consistent] = scale

    segmented = segment_to_grey_levels(selected_smoothed, grey_levels)
    return segmented, scale_map


def naive_segment(reconstruction: np.ndarray, grey_levels: list) -> np.ndarray:
    """Simple per-pixel nearest-grey-level thresholding (original DART)."""
    return segment_to_grey_levels(
        reconstruction.astype(float), np.asarray(grey_levels, dtype=float)
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    H, W = 128, 128
    phantom = np.zeros((H, W))
    phantom[20:100, 20:100] = 128
    phantom[40:80,  40:80]  = 255
    grey_levels = [0, 128, 255]

    noise_levels = [15, 30, 50]
    fig, axes = plt.subplots(len(noise_levels), 5, figsize=(16, 4 * len(noise_levels)))

    for row, sigma_noise in enumerate(noise_levels):
        noisy = phantom + rng.normal(0, sigma_noise, phantom.shape)

        naive_seg          = naive_segment(noisy, grey_levels)
        ms_seg, scale_map  = lidar_multiscale_segment(
            noisy, grey_levels, scales=[1, 3, 5, 7]
        )

        naive_errors = int((naive_seg != phantom).sum())
        ms_errors    = int((ms_seg   != phantom).sum())

        axes[row, 0].imshow(phantom,   cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title("Phantom")
        axes[row, 1].imshow(noisy,     cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Noisy recon (σ={sigma_noise})")
        axes[row, 2].imshow(naive_seg, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title(f"Naive\nerrors={naive_errors}")
        axes[row, 3].imshow(ms_seg,    cmap="gray", vmin=0, vmax=255)
        axes[row, 3].set_title(f"Multiscale (Gaussian)\nerrors={ms_errors}")
        axes[row, 4].imshow(scale_map, cmap="viridis")
        axes[row, 4].set_title("Scale map\n(bright=coarse)")
        for ax in axes[row]:
            ax.axis("off")

        print(f"σ={sigma_noise:2d} | naive: {naive_errors:5d} errors | "
              f"multiscale: {ms_errors:5d} errors | "
              f"improvement: {naive_errors - ms_errors:+d}")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/multiscale_segmentation_demo.png", dpi=150)
    print("Saved demo to multiscale_segmentation_demo.png")