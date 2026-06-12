import astra
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from src.DART.Sinograms import Sinogram

def _projection_order_list(n_angles: int, method: str) -> list:
    """Return a list of projection indices for the given order method."""
    if method == 'interleaved':
        order, seen, queue = [], set(), [(0, n_angles - 1)]
        while len(order) < n_angles:
            next_queue = []
            for lo, hi in queue:
                mid = (lo + hi) // 2
                if mid not in seen:
                    order.append(mid)
                    seen.add(mid)
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

    rec_type = "SART_CUDA" if use_gpu else "SART"
    
    local_sino_id = sino_id
    local_projector_id = projector_id
    allocated_local_sino = False
    allocated_local_proj = False

    if projection_order not in ['random', 'sequential']:
        if proj_geom is None:
            raise ValueError(f"proj_geom must be provided to use '{projection_order}' custom ordering workaround.")

        # 1. Fetch current data and angles from the existing projection geometry
        current_sinogram = astra.data2d.get(sino_id)
        current_angles = np.array(proj_geom['ProjectionAngles'])
        n_angles = current_sinogram.shape[0]

        # 2. Generate the interleaved custom permutation mapping
        order_list = _projection_order_list(n_angles, projection_order)

        # 3. Permute both the sinogram rows and the sequence of projection angles
        reordered_sinogram = current_sinogram[order_list, :]
        reordered_angles = current_angles[order_list]

        # 4. Create a modified local projection geometry copy with the reordered angles
        local_proj_geom = proj_geom.copy()
        local_proj_geom['ProjectionAngles'] = reordered_angles.tolist()

        # 5. Spin up brand new localized ASTRA objects matching this alignment
        local_sino_id = astra.data2d.create('-sino', local_proj_geom, data=reordered_sinogram)
        allocated_local_sino = True

        if not use_gpu:
            local_projector_id = astra.create_projector('linear', local_proj_geom, vol_geom)
            allocated_local_proj = True
        
        # Override tracking behavior to read sequentially now that it is pre-sorted
        actual_projection_order = 'sequential'
    else:
        actual_projection_order = projection_order

    rec_id = astra.data2d.create("-vol", vol_geom, data=vol_data)

    alg_cfg: dict[str, Any] = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = local_projector_id
    alg_cfg["ProjectionDataId"] = local_sino_id
    alg_cfg["ReconstructionDataId"] = rec_id

    # Handle mask: convert to float32 if necessary
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

    # Clean up native core items
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(mask_id)
    
    # Clean up transient data generated specifically for the workaround
    if allocated_local_sino:
        astra.data2d.delete(local_sino_id)
    if allocated_local_proj:
        astra.projector.delete(local_projector_id)

    return reconstruction_img
