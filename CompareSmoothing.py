from src.DART.DART import DART, ParseArgs
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
args = ParseArgs()
phantom_path = os.path.join(f"TestPhantoms", f"phantom_{4}", f"{0}.npy")
phantom = np.load(phantom_path)

reconstruction, results = DART(phantom=phantom,
                                graylevels=np.unique(phantom),
                                p=args.p,
                                dart_iters=args.dart_iters,

                                arm_iters=args.arm_iters,
                                init_arm_iters=args.init_arm_iters,

                                angles=np.linspace(args.lower_angle, args.upper_angle, 25, endpoint=True),
                                detector_spacing=1,
                                n_detectors=512,
                                angle_ordering=args.angle_ordering,
                                
                                relaxation=args.relaxation,
                                vol_data=0,
                                use_gpu=args.gpu,
                                
                                SNR=10,
                                
                                smoothing=None,
                                verbal=args.verbal)

smooth_reconstruction, results = DART(phantom=phantom,
                                graylevels=np.unique(phantom),
                                p=args.p,
                                dart_iters=args.dart_iters,

                                arm_iters=args.arm_iters,
                                init_arm_iters=args.init_arm_iters,

                                angles=np.linspace(args.lower_angle, args.upper_angle, 25, endpoint=True),
                                detector_spacing=1,
                                n_detectors=512,
                                angle_ordering=args.angle_ordering,
                                
                                relaxation=args.relaxation,
                                vol_data=0,
                                use_gpu=args.gpu,
                                
                                SNR=10,
                                
                                smoothing=1,
                                verbal=args.verbal)




fig, ax = plt.subplots(1, 3)
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

ax[0].imshow(phantom, cmap='viridis')
ax[0].set_title("Phantom")
ax[1].imshow(reconstruction, cmap='viridis')
ax[1].set_title(r"$\sigma$=0")
ax[2].imshow(smooth_reconstruction, cmap='viridis')
ax[2].set_title(r"$\sigma$=1")
plt.tight_layout()
plt.savefig("CompareSmoothing4")
# plt.show()

fig, ax = plt.subplots(1, 3)
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

ax[0].imshow(phantom, cmap='gray')
ax[0].set_title("Phantom")
ax[1].imshow(reconstruction, cmap='gray')
ax[1].set_title(r"$\sigma$=0")
ax[2].imshow(smooth_reconstruction, cmap='gray')
ax[2].set_title(r"$\sigma$=1")
plt.tight_layout()
plt.savefig("CompareSmoothing5")
# plt.show()
# sigma controls the width of your kernel (higher = smoother/more blurred density)
kernel_size = 5 

# We apply the filter directly to the 2D images (no flattening!)
filtered_reconstruction = np.zeros_like(reconstruction)
filtered_reconstruction[reconstruction == np.max(phantom)] = 255
recon_density = gaussian_filter(filtered_reconstruction, sigma=kernel_size)
grad = np.gradient(recon_density)
edge = np.sqrt(grad[0]**2 + grad[1]**2)

# Plot as heatmaps
im0 = ax[0].imshow(phantom, cmap='viridis')
ax[0].set_title("Phantom Spatial Density")

im1 = ax[1].imshow(filtered_reconstruction, cmap='viridis')
ax[1].set_title(r"$\sigma$=0 Spatial Density")

im2 = ax[2].imshow(recon_density, cmap='viridis')
ax[2].set_title(r"$\sigma$=1 Spatial Density")

# Add a colorbar to show the density scale
fig.colorbar(im2, ax=ax.ravel().tolist(), shrink=0.6)

plt.savefig("density")