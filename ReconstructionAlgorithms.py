import astra
import numpy as np


def SIRT(vol_geom: dict[str, dict],
         proj_geom: dict,
         sinogram: np.ndarray,
         projector_id: int,
         vol_data: np.ndarray | float = 0,
         iters: int = 200,
         mask: np.ndarray | None = None,
         min_constraint: float = 0,
         max_constraint: float = 255,
         use_gpu: bool = False) -> tuple[int, np.ndarray]:
    
    """
    Perform SIRT (Simultaneous Iterative Reconstruction Technique) reconstruction.

    Parameters
    ----------
    vol_geom : np.ndarray
        Volume geometry as created by `astra.create_vol_geom`.
    proj_geom : np.ndarray
        Projection geometry as created by `astra.create_proj_geom`.
    sinogram : np.ndarray
        2D array containing the sinogram (projection data).
    projector_id : int
        ASTRA projector ID obtained from `astra.create_projector`.
    vol_data : np.ndarray or float, optional
        Initial volume data for the reconstruction. If a scalar is given,
        the volume is initialised with that constant value. Default is 0.
    iters : int, optional
        Number of iterations to run. Default is 200.
    mask : np.ndarray or None, optional
        Optional sinogram mask. If provided, its shape must match the sinogram.
        Default is None.
    min_constraint : float, optional
        Lower bound constraint for pixel values. Default is 0.
    max_constraint : float, optional
        Upper bound constraint for pixel values. Default is 255.
    use_gpu : bool, optional
        If True, use the GPU‑accelerated version (SIRT_CUDA). Default is False.

    Returns
    -------
    tuple[int, np.ndarray]
        A tuple containing:
            rec_id : int
                ASTRA data object ID of the reconstruction volume.
            reconstruction_img : np.ndarray
                2D array of the reconstructed image.
    """

    rec_type = "SIRT_CUDA" if use_gpu else "SIRT"

    sino_id = astra.data2d.create("-sino", proj_geom, data=sinogram)
    rec_id  = astra.data2d.create("-vol",  vol_geom,  data=vol_data)

    alg_cfg = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = projector_id
    alg_cfg["ProjectionDataId"] = sino_id
    alg_cfg["ReconstructionDataId"] = rec_id
    alg_cfg["MinConstraint"] = min_constraint
    alg_cfg["MaxConstraint"] = max_constraint

    mask_id = None
    if mask is not None:
        assert mask.shape == sinogram.shape, (
            f"Mask shape {mask.shape} must match sinogram shape {sinogram.shape}"
        )
        mask_id = astra.data2d.create("-sino", proj_geom, data=mask)
        alg_cfg["SinogramMaskId"] = mask_id

    # Run
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, iters)
    reconstruction_img = astra.data2d.get(rec_id)

    # Clean up ASTRA objects
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(sino_id)
    if mask_id is not None:
        astra.data2d.delete(mask_id)

    return rec_id, reconstruction_img


def FBP(vol_geom: np.ndarray, 
        sino_id: int, 
        proj_geom: np.ndarray,  # Add this argument
        sinogram: np.ndarray,
        use_gpu: bool=False) -> tuple[int, np.ndarray]:

    """
    Perform FBP (Filtered Back Projection) reconstruction.

    Parameters
    ----------
    vol_geom : np.ndarray
        Volume geometry as created by `astra.create_vol_geom`.
    sino_id : int
        ASTRA sinogram data ID (note: this parameter is currently not used
        internally; a new sinogram data object is created from the provided
        sinogram array).
    proj_geom : np.ndarray
        Projection geometry as created by `astra.create_proj_geom`.
    sinogram : np.ndarray
        2D array containing the sinogram (projection data).
    use_gpu : bool, optional
        If True, use the GPU‑accelerated version (FBP_CUDA). Default is False.

    Returns
    -------
    tuple[int, np.ndarray]
        A tuple containing:
            rec_id : int
                ASTRA data object ID of the reconstruction volume.
            reconstruction_img : np.ndarray
                2D array of the reconstructed image.
    """

    sino_id = astra.data2d.create('-sino', proj_geom, data=sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom, data=0)

    # define FBP configuration parameters
    alg_cfg = astra.astra_dict('FBP_CUDA' if use_gpu else 'FBP')
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = rec_id

    if not use_gpu:
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)
        alg_cfg['ProjectorId'] = proj_id

    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id)
    rec = astra.data2d.get(rec_id)

    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(sino_id)
    if not use_gpu:
        astra.projector.delete(proj_id)
    return rec_id, rec