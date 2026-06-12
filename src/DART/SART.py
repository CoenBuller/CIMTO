import astra
import numpy as np

from typing import Any

def _projection_order_list(n_angles: int, method: str) -> list:
    """
    Build a permutation of [0, n_angles) that controls the SART update order.
 
    Parameters
    ----------
    n_angles : int
        Total number of projection angles.
    method : str
        'interleaved' – bisection-based ordering that maximises angular
        separation between consecutive projections (analogous to the
        "maximally separated" scheme).  At each level of the bisection tree,
        the midpoint of every active sub-interval is visited before any child
        interval is opened, so early updates see the widest possible spread of
        angles.
 
    Returns
    -------
    order : list[int]
        Permutation of projection indices in the desired visit order.
    """
    if method == 'interleaved':
        order, seen, queue = [], set(), [(0, n_angles - 1)]
        while len(order) < n_angles:
            next_queue = []
            for lo, hi in queue:
                mid = (lo + hi) // 2
                if mid not in seen:
                    order.append(mid)
                    seen.add(mid)
                # Push left and right sub-intervals for the next BFS level
                if lo < mid:
                    next_queue.append((lo, mid - 1))
                if mid < hi:
                    next_queue.append((mid + 1, hi))
            queue = next_queue
        return order
    else:
        raise ValueError(f"Unknown projection_order '{method}'. "
                         f"Choose from: 'random', 'sequential', 'interleaved'.")

def SART(sino_id: int,
         vol_geom: dict,
         projector_id: int,
         min_constraint: int = 0,
         max_constraint: int = 255,
         relaxation: float = 1.0,
         vol_data: float | np.ndarray = 0,
         iters: int = 200,
         mask=None,
         projection_order: str = 'random',
         use_gpu: bool = False,
         proj_geom: dict | None = None) -> np.ndarray:
    
    """
    Run the Simultaneous Algebraic Reconstruction Technique (SART).
 
    ASTRA only supports 'random' and 'sequential' projection orders natively.
    For any other order (e.g. 'interleaved'), this function works around the
    limitation by:
      1. Computing the desired permutation of angle indices.
      2. Reordering both the sinogram rows and the projection angles
         accordingly.
      3. Creating temporary local ASTRA objects with the reordered data.
      4. Running SART with the 'sequential' order on the pre-sorted data,
         which is equivalent to the custom order.
 
    Parameters
    ----------
    sino_id : int
        ASTRA data ID of the sinogram (or residual sinogram in DART).
    vol_geom : dict
        ASTRA volume geometry.
    projector_id : int
        ASTRA projector ID.
    min_constraint : int
        Lower value clamp applied after each SART update.
    max_constraint : int
        Upper value clamp applied after each SART update.
    relaxation : float
        Step-size multiplier λ.  λ = 1 is the standard SART step; λ < 1 gives
        more conservative (under-relaxed) updates.
    vol_data : float or NDArray
        Initial volume; 0 → zero-initialised.
    iters : int
        Number of SART iterations (each iteration visits all projections once).
    mask : NDArray or None
        Binary reconstruction mask; only pixels where mask=1 are updated.
        None → all pixels are free.
    projection_order : str
        'random' | 'sequential' | 'interleaved' (or any method accepted by
        `_projection_order_list`).
    use_gpu : bool
        Use SART_CUDA instead of the CPU implementation.
    proj_geom : dict or None
        Required when projection_order is not 'random' or 'sequential' so that
        a reordered geometry can be constructed for the workaround.
 
    Returns
    -------
    reconstruction_img : NDArray
        Reconstructed volume after `iters` SART iterations.
    """

    rec_type = "SART_CUDA" if use_gpu else "SART"
    
    # These may be replaced with local copies if a custom ordering workaround
    # is needed; tracking allocation allows correct cleanup.
    local_sino_id = sino_id
    local_projector_id = projector_id
    allocated_local_sino = False
    allocated_local_proj = False

    if projection_order not in ['random', 'sequential']:
        # Custom projection ordering workaround 
        # ASTRA did not support arbitrary ProjectionOrderList in SART, so we
        # pre-permute the sinogram rows and projection angles to match the
        # desired order, then run sequentially on the reordered data.

        if proj_geom is None:
            raise ValueError(f"proj_geom must be provided to use '{projection_order}' custom ordering workaround.")

        # Fetch current data and angles from the existing projection geometry
        current_sinogram = astra.data2d.get(sino_id)
        current_angles = np.array(proj_geom['ProjectionAngles'])
        n_angles = current_sinogram.shape[0]

        # Generate the interleaved custom permutation mapping
        order_list = _projection_order_list(n_angles, projection_order)

        # Permute both the sinogram rows and the sequence of projection angles
        reordered_sinogram = current_sinogram[order_list, :]
        reordered_angles = current_angles[order_list]

        # Create a modified local projection geometry copy with the reordered angles
        local_proj_geom = proj_geom.copy()
        local_proj_geom['ProjectionAngles'] = reordered_angles.tolist()

        # Spin up brand new localized ASTRA objects matching this alignment
        local_sino_id = astra.data2d.create('-sino', local_proj_geom, data=reordered_sinogram)
        allocated_local_sino = True

        if not use_gpu:
            local_projector_id = astra.create_projector('linear', local_proj_geom, vol_geom)
            allocated_local_proj = True
        
        # Override tracking behavior to read sequentially now that it is pre-sorted
        actual_projection_order = 'sequential'
    else:
        actual_projection_order = projection_order


    # Build and run SART
    rec_id = astra.data2d.create("-vol", vol_geom, data=vol_data)

    alg_cfg: dict[str, Any] = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = local_projector_id
    alg_cfg["ProjectionDataId"] = local_sino_id
    alg_cfg["ReconstructionDataId"] = rec_id

    # Handle mask
    if mask is None:
        mask = np.ones((vol_geom['GridRowCount'], vol_geom['GridColCount']), dtype=np.float32)
    else:
        mask = mask.astype(np.float32)
    mask_id = astra.data2d.create('-vol', vol_geom, mask)

    # Build options dictionary
    options = {
        'ReconstructionMaskId': mask_id,
        'MaxConstraint': max_constraint,
        'MinConstraint': min_constraint,
        'Relaxation': float(relaxation),
        'ProjectionOrder': actual_projection_order
    }
    alg_cfg['option'] = options

    # Run algorithm
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.astra_dict
    astra.algorithm.run(algorithm_id, iters)
    reconstruction_img = astra.data2d.get(rec_id)

    # Clean up
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(mask_id)
    
    if allocated_local_sino:
        astra.data2d.delete(local_sino_id)
    if allocated_local_proj:
        astra.projector.delete(local_projector_id)

    return reconstruction_img
