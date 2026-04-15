import numpy as np
import astra
import matplotlib.pyplot as plt

from os.path import isdir
from os import mkdir
from pathlib import Path
from PIL import Image
from skimage import draw
from ReconstructionAlgorithms import SIRT, FBP

def sinogram(phantom: np.ndarray, 
            n_detectors: int, 
            angles: np.ndarray, 
            detector_spacing: int, 
            beam_type: str='parallel',
            intensity_scale: int | None=None, 
            save_dir: str | None=None, 
            n_projections: int | None=None,
            use_gpu: bool=False):
        
    """
    Generate sinograms from a phantom and optionally add Poisson noise.

    Parameters
    ----------
    phantom : np.ndarray
        2D array representing the object (phantom) to be projected.
    n_detectors : int
        Number of detector elements in the projection geometry.
    angles : np.ndarray
        Array of projection angles (in radians) over which the sinogram is computed.
    detector_spacing : int
        Spacing between detector elements.
    beam_type : str, optional
        Type of beam geometry. Must be either 'parallel' or 'fanflat'.
        Default is 'parallel'.
    intensity_scale : int or None, optional
        If provided, Poisson noise is added to the sinogram using this value as
        the mean photon count I0. Higher values produce higher signal‑to‑noise ratio.
        Default is None (no noise added).
    save_dir : str or None, optional
        Directory path where individual projection images will be saved.
        If provided and does not exist, it will be created. Default is None.
    n_projections : int or None, optional
        Number of projections (used only when saving images to name the files).
        If `save_dir` is provided, this should be the length of the `angles` array.
        Default is None.
    use_gpu : bool, optional
        If True, use the GPU‑accelerated projector. Default is False.

    Returns
    -------
    tuple
        Depending on whether `intensity_scale` is provided, the return tuple is:
        
        If `intensity_scale` is None:
            (proj_id, sino_id, sinogram, vol_geom, proj_geom)
        If `intensity_scale` is not None:
            (proj_id, sino_id, sinogram_noisy, vol_geom, proj_geom)
        
        Where:
            proj_id : int
                ASTRA projector ID.
            sino_id : int
                ASTRA sinogram data ID (contains the final sinogram).
            sinogram / sinogram_noisy : np.ndarray
                The computed sinogram (2D array).
            vol_geom : np.ndarray
                Volume geometry derived from the phantom shape.
            proj_geom : np.ndarray
                Projection geometry created with the given beam parameters.
    """

    width, height = phantom.shape
    vol_geom = astra.creators.create_vol_geom([width,height])
    phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)

    if beam_type not in ['parallel', 'fanflat']:
        raise ValueError("beam type must be either 'parallel' or 'fanflat'")

    # create projection geometry
    proj_geom = astra.create_proj_geom(beam_type, detector_spacing, n_detectors, angles)

    # choose projector
    if use_gpu:
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    else:
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

    # Create sinogram
    sino_id, sinogram = astra.creators.create_sino(phantom_id, proj_id)

    # Apply Poisson noise.
    if intensity_scale is not None:
        photon_counts = intensity_scale * np.exp(-sinogram/np.max(sinogram))
        noisy_counts = np.random.poisson(photon_counts).astype(np.float64)
        noisy_counts[noisy_counts <= 0] = 1

        # Convert back to attenuation space (line integrals)
        sinogram_noisy = -np.log(noisy_counts / intensity_scale)
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram_noisy)
    
    # Save projections as images, if directory has been defined.
    if save_dir != None:
        if save_dir[-1] != '/':
            save_dir += '/'
        if not isdir(save_dir):
            mkdir(save_dir)
        proj_for_img = np.round(sinogram * (2**8- 1)).astype(np.uint8)
        for i in range(n_projections):
            Image.fromarray(proj_for_img[i]).save(save_dir+f'proj_{i}.png')
    
    if intensity_scale is not None:
        return proj_id, sino_id, sinogram_noisy, vol_geom, proj_geom

    return proj_id, sino_id, sinogram, vol_geom, proj_geom


if __name__ == "__main__":


    """Demonstration of how the sinogram generator and SIRT and FBP functions can be used"""
    IMG_SHAPE =(512, 512)
    from scipy.ndimage import gaussian_filter
    
    noise = np.random.standard_normal(IMG_SHAPE)
    smooth = gaussian_filter(noise, sigma=2)
    clippedSmooth = np.clip(smooth, 0, 1)

    # Make the image binary
    threshold = 0.04
    clippedSmooth[clippedSmooth > threshold] = 1
    clippedSmooth[clippedSmooth <= threshold] = 0

    # Define circular field
    radius = 240
    center = np.array(IMG_SHAPE) // 2
    rr, cc = draw.ellipse(center[0], center[1], radius, radius, shape=IMG_SHAPE)

    # Use circular field to define a mask
    mask = np.ones(IMG_SHAPE)
    mask[rr, cc] = 0

    # Set everything outside the circle equal to 0
    clippedSmooth[mask.astype(bool)] = 0

    clippedSmooth *= 255
    clippedSmooth[clippedSmooth > 255] = 255
    clippedSmooth[clippedSmooth < 0] = 0



    proj_id, sino_id, sinogram_img, volume_geom, projection_geom = sinogram(phantom=clippedSmooth,
                                                                            n_detectors=512,
                                                                            angles=np.linspace(0, np.pi, 180),
                                                                            detector_spacing=1)
    
    rec_id, sirt_reconstruction = SIRT(vol_geom=volume_geom, 
                                      proj_geom=projection_geom,
                                      sinogram=sinogram_img,
                                      projector_id=proj_id,
                                      vol_data=0,
                                      iters=200)
    
    rec_id, fbp_reconstruction = FBP(vol_geom=volume_geom,
                                     proj_geom=projection_geom,
                                     sinogram=sinogram_img,
                                     sino_id=sino_id,
                                     use_gpu=False)
    
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(clippedSmooth, cmap='gray')
    ax[1].imshow(sinogram_img)
    ax[2].imshow(sirt_reconstruction)
    ax[3].imshow(fbp_reconstruction)
    plt.show()