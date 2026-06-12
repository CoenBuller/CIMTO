import numpy as np
import astra

from typing import Callable
from numpy.typing import NDArray



def PoissonNoise(sinogram: NDArray, SNR: int) -> NDArray:
    """
    Add physically motivated Poisson noise to a sinogram (Beer-Lambert model).
 
    The sinogram is treated as line-integrated attenuation values
    (i.e. -log(I/I0)).  The procedure is:
 
      1. Normalise the sinogram to a mean attenuation of 1 so that the SNR
         target is resolution-independent.
      2. Back-calculate the incident intensity I0 that would produce the
         requested SNR, defined here as SNR = sqrt(I0 * E[exp(-A_norm)]),
         which gives SNR² = I0 * mean(exp(-A_norm)).
      3. Sample Poisson-distributed photon counts around the expected
         transmitted intensity I0 * exp(-A_norm).
      4. Convert the noisy counts back to attenuation values and rescale to
         the original sinogram magnitude.
 
    Parameters
    ----------
    sinogram : NDArray
        Clean sinogram (line integrals / attenuation projections).
    SNR : int
        Target signal-to-noise ratio.  Higher values → less noise.
 
    Returns
    -------
    sinogram_noisy : NDArray
        Noise-corrupted sinogram with the same shape and scale as the input.
    """
 
    A_mean = sinogram.mean()        # Scale factor for un-normalising at the end
    A_norm = sinogram / A_mean      # Normalise so mean attenuation = 1
 
    # I0 satisfying SNR² = I0 * mean(exp(-A_norm))
    I0 = SNR ** 2 / np.exp(-A_norm).mean()
 
    # Expected transmitted intensity at each detector bin
    measured_intensity = I0 * np.exp(-A_norm)
 
    # Draw Poisson counts; clip to 1 to avoid log(0) in the back-conversion
    noisy_counts = np.random.poisson(measured_intensity).astype(np.float64)
    noisy_counts = np.maximum(noisy_counts, 1)
 
    # Convert counts back to attenuation and restore original scale
    sinogram_noisy = A_mean * -np.log(noisy_counts / I0)
    return sinogram_noisy

def Sinogram(phantom: np.ndarray, 
            n_detectors: int, 
            angles: np.ndarray, 
            detector_spacing: int, 
            beam_type: str='parallel',
            SNR: int | None=None, 
            noise_func: Callable=PoissonNoise,
            n_projections: int = 0,
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
    SNR: int or None, optional
        If provided, Poisson noise is added to the sinogram using this value as
        the mean photon count I0. Higher values produce higher signal‑to‑noise ratio.
        Medium amount of noise added is done by setting I0 = 10e3
        Default is None (no noise added).
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


    if beam_type not in ['parallel', 'fanflat']:
        raise ValueError("beam type must be either 'parallel' or 'fanflat'")
    
    if noise_func not in [PoissonNoise]:
        raise ValueError("Noise type must be either PoissonNoise, GuassianNoise or 'uniform'")

    sinogram_noisy = None
    width, height = phantom.shape
    vol_geom = astra.creators.create_vol_geom([width,height])
    phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)


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
    if SNR is not None:
        sinogram_noisy = noise_func(sinogram, SNR)
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram_noisy)
    
    astra.data2d.delete(phantom_id)

    if SNR is not None:
        assert sinogram_noisy is not None, "The noisy sinogram is still None but should be a np.ndarray."
        return proj_id, sino_id, sinogram_noisy, vol_geom, proj_geom
    

    return proj_id, sino_id, sinogram, vol_geom, proj_geom


def ResidualSinogram(reconstruction: np.ndarray,
                     free_mask: np.ndarray,
                     sinogram_img: np.ndarray,
                     projector_id: int,
                     vol_geom: dict[str, dict],
                     projector_geom: dict[str, dict]) -> int:
    """
    Compute the residual sinogram b_res = b_0 - A(x_fixed).
 
    Only the fixed pixels (where free_mask is False) contribute to the
    forward projection A(x_fixed).  The residual captures the portion of
    the sinogram that still needs to be explained by the free pixels, and
    is passed directly to SART in the ARM step.
 
    Parameters
    ----------
    reconstruction : np.ndarray
        Current (segmented) reconstruction containing both fixed and free pixels.
    free_mask : np.ndarray
        Boolean mask; True where a pixel is free (will be reconstructed).
    sinogram_img : np.ndarray
        The measured (target) sinogram b_0.
    projector_id : int
        ASTRA projector ID.
    vol_geom : dict
        ASTRA volume geometry.
    projector_geom : dict
        ASTRA projection geometry.
 
    Returns
    -------
    residual_sino_id : int
        ASTRA data ID for the residual sinogram.
    """
 
    # Zero out free pixels so only fixed pixels contribute to the projection
    fixed_only = reconstruction.copy()
    fixed_only[free_mask] = 0
 
    fixed_phantom_id = astra.data2d.create('-vol', vol_geom, fixed_only)
    fixed_sino_id, fixed_sino = astra.creators.create_sino(fixed_phantom_id, projector_id)
    astra.data2d.delete(fixed_phantom_id)
 
    # Subtract the fixed contribution from the measured sinogram
    residual_sino = sinogram_img - astra.data2d.get(fixed_sino_id)
    residual_sino_id = astra.data2d.create('-sino', projector_geom, residual_sino)
    astra.data2d.delete(fixed_sino_id)
 
    return residual_sino_id


