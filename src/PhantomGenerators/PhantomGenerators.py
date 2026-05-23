import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage import draw


def RectangleDist(height, width, theta, rot=0.):
    """
    Compute the distance from the center to the edge of a rectangle for each given angle.

    Parameters
    ----------
    height : float
        The height of the rectangle (along the row direction).
    width : float
        The width of the rectangle (along the column direction).
    theta : 1D numpy array
        Array of angles (in radians) at which to compute the distance.
    rot : float, optional
        Rotation angle of the rectangle (in radians). Default 0.

    Returns
    -------
    r : 1D numpy array
        Radial distance from the center to the rectangle edge for each angle.
    """
    w = 2 * np.abs(np.cos(theta + rot)) / width
    h = 2 * np.abs(np.sin(theta + rot)) / height
    wh = np.concat([w[:, np.newaxis], h[:, np.newaxis]], axis=1)
    r = 1 / (np.max(wh, axis=1))
    return r


def EllipseDist(b, a, theta, rot=0.):
    """
    Compute the distance from the center to the edge of an ellipse for each given angle.

    Parameters
    ----------
    b : float
        Semi-minor axis (or axis along the row direction, depending on convention).
    a : float
        Semi-major axis (or axis along the column direction).
    theta : 1D numpy array
        Array of angles (in radians) at which to compute the distance.
    rot : float, optional
        Rotation angle of the ellipse (in radians). Default 0.

    Returns
    -------
    r : 1D numpy array
        Radial distance from the center to the ellipse edge for each angle.
    """
    r = (a * b) / np.sqrt((b * np.cos(theta + rot))**2 + (a * np.sin(theta + rot))**2)
    return r


def CreateBlob(center, width, height, radiusFunction, noise_amplitude, sigma,
               num_points=200, rot=0., theta=None, rng=None):
    """
    Generate the row and column coordinates of a blob (closed contour) with optional
    smooth noise applied to the radius.

    Parameters
    ----------
    center : tuple of float
        (row_center, col_center) coordinates of the blob's center.
    width : float
        Width parameter for the shape (interpreted by radiusFunction).
    height : float
        Height parameter for the shape (interpreted by radiusFunction).
    radiusFunction : callable or numpy.ndarray
        If callable, it must accept arguments (width, height, angles, rot) and return
        a radius array. If an array, it is treated as pre‑computed radii (must match
        the number of angles).
    noise_amplitude : float
        Standard deviation of the Gaussian noise added to the radius before smoothing.
        If zero, no noise is added.
    sigma : float
        Standard deviation of the Gaussian filter applied to the noise (smoothing).
    num_points : int, optional
        Number of angular samples if theta is not provided. Default 200.
    rot : float, optional
        Rotation angle (in radians) passed to radiusFunction. Default 0.
    theta : array-like or None, optional
        Explicit angles (in radians) at which to sample. If provided, num_points is ignored.
        Default None.

    Returns
    -------
    r_coords : 1D numpy array
        Row coordinates of the blob boundary.
    c_coords : 1D numpy array
        Column coordinates of the blob boundary.
    """

    if rng is None:
        rng = np.random.default_rng(67)

    # Generate angles if not provided explicitly
    if theta is None:
        angles = np.linspace(0, 2 * np.pi, num_points)
    else:
        angles = np.asarray(theta)
        num_points = len(angles)

    # Generate noise if required
    if noise_amplitude != 0:
        noise = rng.normal(0, noise_amplitude, num_points)
        smooth_noise = gaussian_filter1d(noise, sigma=sigma, mode='wrap')
    else:
        smooth_noise = np.zeros(num_points)

    # Compute the base radii either from a function or a precomputed array
    if isinstance(radiusFunction, np.ndarray):
        radii = radiusFunction + smooth_noise
    else:
        radii = radiusFunction(width, height, angles, rot=rot) + smooth_noise

    # Convert polar coordinates (angles, radii) to Cartesian (row, column)
    c_coords = center[1] + radii * np.cos(angles)
    r_coords = center[0] + radii * np.sin(angles)

    return r_coords, c_coords


def ShrinkShape(r_coords, c_coords, shrinkFactor, smoothing=3, rot=0., rng=None):
    """
    Shrink a shape defined by its boundary coordinates towards its centroid,
    apply smoothing, and optionally rotate it.

    Parameters
    ----------
    r_coords : 1D numpy array
        Row coordinates of the original boundary points.
    c_coords : 1D numpy array
        Column coordinates of the original boundary points.
    shrinkFactor : float
        Factor by which to shrink the shape (e.g., 0.5 makes it half the size).
    smoothing : int, optional
        Standard deviation of the Gaussian filter applied to the shrunk coordinates.
        Default 3.
    rot : float, optional
        Rotation angle (in radians) to apply after shrinking and smoothing.
        Default 0 (no rotation).

    Returns
    -------
    new_r : 1D numpy array
        Row coordinates of the processed shape.
    new_c : 1D numpy array
        Column coordinates of the processed shape.
    """
    # Compute centroid of the original shape
    midR, midC = np.mean(r_coords), np.mean(c_coords)

    # Compute angles of each point relative to centroid, then sort by angle
    angles = np.arctan2(c_coords - midC, r_coords - midR)
    sort_idx = np.argsort(angles)
    rr_sorted = r_coords[sort_idx]
    cc_sorted = c_coords[sort_idx]

    # Shift to origin, shrink, then shift back (but we keep them centered temporarily)
    r_centered = (rr_sorted - midR) * shrinkFactor
    c_centered = (cc_sorted - midC) * shrinkFactor

    # Smooth the coordinates independently
    if smoothing != 0:
        smoothR = gaussian_filter1d(r_centered, smoothing)
        smoothC = gaussian_filter1d(c_centered, smoothing)
    else:
        smoothR, smoothC = r_centered, c_centered

    # Apply optional rotation around the origin
    if rot != 0:
        rotC = np.cos(rot) * smoothC - np.sin(rot) * smoothR
        rotR = np.sin(rot) * smoothC + np.cos(rot) * smoothR
    else:
        rotR, rotC = smoothR, smoothC

    # Shift back to the original centroid
    return rotR + midR, rotC + midC


def EdgeImage(img):
    """
    Extract the coordinates of all edge pixels in an image.
    Edge pixels are defined as those where the gradient magnitude is non‑zero.

    Parameters
    ----------
    img : 2D numpy array
        Input image (grayscale).

    Returns
    -------
    r : 1D numpy array
        Row indices of edge pixels.
    c : 1D numpy array
        Column indices of edge pixels.
    """
    grad = np.gradient(img)
    edgeImg = np.sqrt(grad[0]**2 + grad[1]**2)
    r, c = np.where(edgeImg != 0)
    return r, c


def CartesianToPolar(r, c):
    """
    Convert a set of Cartesian (row, column) coordinates to polar coordinates
    relative to their centroid.

    Parameters
    ----------
    r : 1D numpy array
        Row coordinates.
    c : 1D numpy array
        Column coordinates.
    returnCenter : Bool, optional
        If set to True, will also return the center coordinates of the shape

    Returns
    -------
    radius : 1D numpy array
        Radial distance from the centroid to each point.
    theta : 1D numpy array
        Angle (in radians) from the positive x‑axis (column direction) to each point,
        using the convention of arctan2(dy, dx) where dy = row - center_row,
        dx = column - center_col.
    """
    # Compute centroid of the points
    centerR, centerC = np.mean(r), np.mean(c)

    # Shift to origin
    dr = r - centerR
    dc = c - centerC

    # Compute radius and angle
    radius = np.sqrt(dr**2 + dc**2)
    theta = np.arctan2(dr, dc)  # arctan2(y, x) → y = dr, x = dc
    
    return radius, theta

def DifferentGrayLevels(img, levels):
    """Generates a shape in the image with different grayscale values. Works best for convex/geometrical shapes

    Parameters
    ----------
    img : 2D numpy array
        Original binary image with the shape

    levels : int
        Integer determining how many different gray scale values that will appear in the image. Background grayscale value 
        is considered as one level aswell.
    
    Returns
    -------
    imgFinal : 2D numpy arra
        new image with the shape that contains different gray scale values. 

    """


    imgFinal = img.copy() 
    center = (img.shape[0]//2, img.shape[1]//2) # Center of shape

    for i in range(levels):
        shrinkFactor = 1 - (i)/(levels - 1)

        # Edge of original shape
        r, c = EdgeImage(img)
        shrinkR, shrinkC = ShrinkShape(r, c, shrinkFactor=shrinkFactor, smoothing=1, rot=0)
        radii, angles = CartesianToPolar(shrinkR, shrinkC)
        r, c = CreateBlob(center=center, width=0, height=0, radiusFunction=radii, noise_amplitude=0, sigma=1, rot=0, theta=angles)
        rr, cc = draw.polygon(r, c, shape=img.shape)
        imgFinal[rr, cc] = shrinkFactor * np.max(img)

    return imgFinal

def DifferentGrayDistBased(center, img, levels, inverted=False):
    """
    Generate an image with concentric grayscale bands based on distance from a specified center.

    This function divides a shape in the image into `levels` regions according to the distance 
    from a given center point. Each region is assigned a different grayscale intensity, producing
    a gradient-like effect from the center outwards (or inwards if `inverted=True`).

    Parameters
    ----------
    center : tuple of float
        Coordinates (row, column) of the reference center from which distances are computed.
        This is typically the center of the shape or area of interest.
    img : 2D numpy array
        Input image containing the shape. Non-zero pixels are considered part of the shape, 
        while zero pixels are treated as background.
    levels : int
        Number of grayscale levels to generate. The function will divide the distance from 
        the center into `levels` intervals. Higher values create finer gradations.
    inverted : bool, optional
        If True, the grayscale values are inverted (lighter at the edges and darker at the center). 
        Default is False, meaning darker at the edges and lighter at the center.

    Returns
    -------
    imgOut : 2D numpy array
        New image of the same shape as `img`, where pixels inside the original shape are 
        assigned grayscale values depending on their distance from `center`. Background pixels 
        (originally zero) remain zero.

    Notes
    -----
    - The distance-based grayscale is normalized such that the maximum grayscale value 
      corresponds to the maximum pixel value in the input image (`np.max(img)`).
    - This function works best for convex or roughly circular shapes.
    - The output can be used for visualization, synthetic dataset generation, or as a mask
      with graded intensity.
    """
    imgCopy = img.copy()
    
    rows, cols = img.shape
    r = np.arange(rows)
    c = np.arange(cols)
    R, C = np.meshgrid(r, c, indexing="ij")

    # distance to center
    dist = np.sqrt((R - center[0])**2 + (C - center[1])**2)

    # normalize distances
    dist = dist / dist.max()

    # output image
    imgOut = np.zeros_like(imgCopy, dtype=float)

    for i in range(levels):
        lower = i / levels
        upper = (i + 1) / levels

        mask = (dist >= lower) & (dist < upper) & (imgCopy != 0)

        color = 1- i/(levels-1)
        if inverted:
            color =  (1+i)/(levels-1)   # from white to black
        imgOut[mask] = color

    return imgOut * np.max(imgCopy)


