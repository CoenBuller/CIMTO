import astra 
import numpy as np
from typing import Any

def _projection_order_list(n_angles: int, method: str) -> np.ndarray:
    if method == 'sequential':
        return np.arange(n_angles)
    elif method == 'interleaved':
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
        return np.array(order)
    else:
        raise ValueError(f"Unknown projection_order '{method}'. "
                         f"Choose from: 'random', 'reversed', 'interleaved'.")


def SART(sino_id: int,
         vol_geom: dict[str, dict],
         projector_id: int,
        
         min_constraint: int = 0,
         max_constraint: int = 255,
        
         relaxation: float = 1,
         vol_data: float | np.ndarray = 0,
         iters: int = 200,
         mask= None,
         projection_order: str = 'random',
         use_gpu: bool = False) -> np.ndarray:

    rec_type = "SART_CUDA" if use_gpu else "SART"

    rec_id  = astra.data2d.create("-vol",  vol_geom,  data=vol_data)
    alg_cfg: dict[str, Any] = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = projector_id 
    alg_cfg["ProjectionDataId"] = sino_id
    alg_cfg["ReconstructionDataId"] = rec_id

    if mask is None:
        mask = np.ones((vol_geom['GridRowCount'], vol_geom['GridColCount']))
             
    mask_id = astra.data2d.create('-vol', vol_geom, mask)

    if projection_order == 'random':
        alg_cfg['option'] = {
            'ReconstructionMaskId': mask_id,
            'MaxConstraint': max_constraint,
            'MinConstraint': min_constraint,
            'ProjectionOrder': 'random',
            'Relaxation': relaxation
        }
    else:
        n_angles = astra.data2d.get(sino_id).shape[0]
        order_list = _projection_order_list(n_angles, projection_order)
        alg_cfg['option'] = {
            'ReconstructionMaskId': mask_id,
            'MaxConstraint': max_constraint,
            'MinConstraint': min_constraint,
            'ProjectionOrder': 'custom',
            'ProjectionOrderList': order_list,
            'Relaxation': relaxation
        }

    # Run
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, iters)
    reconstruction_img = astra.data2d.get(rec_id)

    # Clean up ASTRA objects
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(rec_id)      
    astra.data2d.delete(mask_id)

    return reconstruction_img

def SIRT(sino_id: int,
         vol_geom: dict[str, dict],
         projector_id: int,
        
         min_constraint: int = 0,
         max_constraint: int = 255,

         vol_data: float | np.ndarray = 0,
         iters: int = 200,
         mask= None,
         use_gpu: bool = False) -> np.ndarray:

    rec_type = "SIRT_CUDA" if use_gpu else "SIRT"

    rec_id  = astra.data2d.create("-vol",  vol_geom,  data=vol_data)
    alg_cfg: dict[str, Any] = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = projector_id 
    alg_cfg["ProjectionDataId"] = sino_id
    alg_cfg["ReconstructionDataId"] = rec_id

    if mask is None:
        mask = np.ones((vol_geom['GridRowCount'], vol_geom['GridColCount']))
    mask_id = astra.data2d.create('-vol', vol_geom, mask)
    alg_cfg['option'] = {
        'ReconstructionMaskId': mask_id,
        'MaxConstraint': max_constraint,
        'MinConstraint': min_constraint
    }
    # Run
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, iters)
    reconstruction_img = astra.data2d.get(rec_id)

    # Clean up ASTRA objects
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(rec_id)      
    astra.data2d.delete(mask_id)

    return reconstruction_img


def FBP(vol_geom: dict[str, dict], 
        sino_id: int, 
        proj_geom: dict[str, dict],
        sinogram: np.ndarray,
        use_gpu: bool=False) -> tuple[int, np.ndarray]:

        sino_id = astra.data2d.create('-sino', proj_geom, data=sinogram)
        rec_id = astra.data2d.create('-vol', vol_geom, data=0)

        # define FBP configuration parameters
        alg_cfg: dict[str, Any] = astra.astra_dict('FBP_CUDA' if use_gpu else 'FBP')
        alg_cfg['ProjectionDataId'] = sino_id #
        alg_cfg['ReconstructionDataId'] = rec_id

        if not use_gpu:
            proj_id = astra.create_projector('linear', proj_geom, vol_geom)
            alg_cfg['ProjectorId'] = proj_id

        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id)
        rec = astra.data2d.get(rec_id)

        astra.algorithm.delete(algorithm_id)
        astra.data2d.delete(sino_id)
        return rec_id, rec