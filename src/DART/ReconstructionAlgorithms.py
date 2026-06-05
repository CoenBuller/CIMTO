import astra 
import numpy as np
from typing import Any

def GenerateAngles(angle_order: str, n: int):
    """Generate projection angles according to the chosen ordering strategy."""

    if angle_order == "sequential":
        return np.linspace(0, np.pi, n, endpoint=False)

    elif angle_order == "maximally_separated":
        # Golden-ratio sampling: each new angle maximally separates from previous
        golden = np.pi * (3.0 - np.sqrt(5.0))
        angles = (np.arange(n) * golden) % (stop - start) + start
        return np.sort(angles)

    else:
        return
def SART(sino_id: int,
         vol_geom: dict[str, dict],
         projector_id: int,
        
         img_shape: tuple[int, int] = (512, 512),
         min_constraint: int = 0,
         max_constraint: int = 255,

         vol_data: float | np.ndarray = 0,
         iters: int = 200,
         relaxation: float = 1.0,
         
         n_projections: int = 10,
         angle_ordering: str = "randomized",
         mask=None,
         use_gpu: bool = False,) -> np.ndarray:

    rec_type = "SART_CUDA" if use_gpu else "SART"

    rec_id = astra.data2d.create("-vol", vol_geom, data=vol_data)
    alg_cfg: dict[str, Any] = astra.astra_dict(rec_type)
    alg_cfg["ProjectorId"] = projector_id
    alg_cfg["ProjectionDataId"] = sino_id
    alg_cfg["ReconstructionDataId"] = rec_id

    if mask is None:
        mask = np.ones(img_shape)

    if angle_ordering != "randomized":
        projection_order = "custom"
        ordering_list = GenerateAngles(angle_ordering, n=n_projections)
    else:
        projection_order = "random"

    mask_id = astra.data2d.create('-vol', vol_geom, mask)
    alg_cfg['option'] = {
        'ReconstructionMaskId': mask_id,
        'MaxConstraint': max_constraint,
        'MinConstraint': min_constraint,
        'Relaxation': relaxation,
        'ProjectionOrder': projection_order,           
    }

    if projection_order == "custom":
        alg_cfg['option']['ProjectionOrderList'] = ordering_list


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
        
         img_shape: tuple[int, int] = (512, 512),
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
        mask = np.ones(img_shape)        
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