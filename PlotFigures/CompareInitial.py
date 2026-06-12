from src.DART.DART import DART, ParseArgs
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
args = ParseArgs()
phantom_path = os.path.join(f"TestPhantoms", f"phantom_{3}", f"{0}.npy")
phantom = np.load(phantom_path)

reconstruction, results = DART(phantom=phantom,
                                graylevels=np.unique(phantom),
                                p=args.p,
                                dart_iters=0,

                                arm_iters=args.arm_iters,
                                init_arm_iters=100,

                                angles=np.linspace(args.lower_angle, args.upper_angle, 25, endpoint=True),
                                detector_spacing=1,
                                n_detectors=512,
                                angle_ordering=args.angle_ordering,
                                
                                relaxation=args.relaxation,
                                vol_data=0,
                                use_gpu=args.gpu,
                                
                                SNR=args.snr,
                                
                                smoothing=args.smoothing,
                                verbal=args.verbal)

smooth_reconstruction, results = DART(phantom=phantom,
                                graylevels=np.unique(phantom),
                                p=args.p,
                                dart_iters=0,

                                arm_iters=args.arm_iters,
                                init_arm_iters=500,

                                angles=np.linspace(args.lower_angle, args.upper_angle, 25, endpoint=True),
                                detector_spacing=1,
                                n_detectors=512,
                                angle_ordering=args.angle_ordering,
                                
                                relaxation=args.relaxation,
                                vol_data=0,
                                use_gpu=args.gpu,
                                
                                SNR=args.snr,
                                
                                smoothing=args.smoothing,
                                verbal=args.verbal)

extra_smooth, results = DART(phantom=phantom,
                                graylevels=np.unique(phantom),
                                p=args.p,
                                dart_iters=0,

                                arm_iters=args.arm_iters,
                                init_arm_iters=2000,

                                angles=np.linspace(args.lower_angle, args.upper_angle, 25, endpoint=True),
                                detector_spacing=1,
                                n_detectors=512,
                                angle_ordering=args.angle_ordering,
                                
                                relaxation=args.relaxation,
                                vol_data=0,
                                use_gpu=args.gpu,
                                
                                SNR=args.snr,
                                
                                smoothing=args.smoothing,
                                verbal=args.verbal)



fig, ax = plt.subplots(1, 3)
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")

ax[0].imshow(phantom, cmap='viridis')
ax[0].set_title(r"$t_{0}$=1")
ax[1].imshow(smooth_reconstruction, cmap='viridis')
ax[1].set_title(r"$t_{0}$=50")
ax[2].imshow(extra_smooth, cmap='viridis')
ax[2].set_title(r"$t_{0}$=200")
plt.tight_layout()
plt.savefig("Comparet0_phantom1")
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
plt.savefig("Comparet0_phantom1_2")
# plt.show()