import os
import numpy as np
import matplotlib.pyplot as plt

from src.DART.DART import DART, ParseArgs

"""Makes 3 reconstructions using DART for a specified phantom class and instance under noise conditions with an SNR=20, and a different smoothing factor for each reconstruction
(s.t. 0 -> 1 -> 3). Smoothing parameters can be changed if needed"""

args = ParseArgs()
phantom_path = os.path.join(f"TestPhantoms", f"phantom_{args.type}", f"{args.instance}.npy")
phantom = np.load(phantom_path)

# First reconstruction, no smoothing
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
                                
                                SNR=20,
                                
                                smoothing=None,
                                verbal=args.verbal)

# Second reconstruction, medium smoothing
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
                                
                                SNR=20,
                                
                                smoothing=1,
                                verbal=args.verbal)

# Thired reconstruction, heavy smoothing
extra_smooth, results = DART(phantom=phantom,
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
                                
                                SNR=20,
                                
                                smoothing=3,
                                verbal=args.verbal)


# Plots all 3 reconstructions next to each other. It will plot it in two colormaps: viridis and gray.
fig, ax = plt.subplots(1, 3)
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

ax[0].imshow(reconstruction, cmap='viridis')
ax[0].set_title(r"$\sigma$=0")
ax[1].imshow(smooth_reconstruction, cmap='viridis')
ax[1].set_title(r"$\sigma$=1")
ax[2].imshow(extra_smooth, cmap='viridis')
ax[2].set_title(r"$\sigma$=3")
plt.tight_layout()
plt.savefig(f"Compare_smoothing_phantom_{args.type}")

fig, ax = plt.subplots(1, 3)
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

ax[0].imshow(phantom, cmap='gray')
ax[0].set_title(r"$\sigma$=0")
ax[1].imshow(reconstruction, cmap='gray')
ax[1].set_title(r"$\sigma$=1")
ax[2].imshow(smooth_reconstruction, cmap='gray')
ax[2].set_title(r"$\sigma$=3")
plt.tight_layout()
plt.savefig(f"Compare_smoothing_phantom_{args.type}_gray")

