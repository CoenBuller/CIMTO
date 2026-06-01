import matplotlib.pyplot as plt
import numpy as np
import astra as at

from Segmentation import discrete_vote_bilateral
from RoundTo import ScaleTo, RoundTo
from ReconstructionAlgorithms import SIRT
from Sinograms import Sinogram

phantom = np.load("PhantomGenerators/Pixel_phantom/phantom_arrays/granular_phantom_single_grayvalues.npy")

# Optimal projection settings
n_projections = 180
n_detectors = 512
angles = np.linspace(0, np.pi, n_projections)
detector_spacing = 1

gray_values = np.unique(phantom)
proj_id, sino_id, sino_data, vol_geom, proj_geom = Sinogram(
                                                            phantom=phantom,
                                                            n_projections=n_projections,
                                                            n_detectors=n_detectors,
                                                            angles=angles,
                                                            detector_spacing=detector_spacing,
                                                            I0=10**2
                                                            )

noisy_res = SIRT(vol_geom=vol_geom, sino_id=sino_id, projector_id=proj_id, vol_data=0, iters=100, use_gpu=False)
noisy_res = ScaleTo(noisy_res)

segmented = discrete_vote_bilateral(noisy_res, gray_values=gray_values, sigma_s=1, sigma_r=np.mean(gray_values)/3)
naive = RoundTo(noisy_res, graylevels=gray_values)

print(f"K wrong pixels for naive: {np.sum(phantom != naive)}")
print(f"K wrong pixels for segmentation: {np.sum(phantom != segmented)}")

fig, ax = plt.subplots(1, 3)
ax[0].imshow(naive, cmap='gray')
ax[1].imshow(segmented, cmap='gray')
ax[2].imshow(noisy_res, cmap='gray')

plt.savefig("jekkr")

