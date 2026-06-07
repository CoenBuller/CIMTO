import matplotlib.pyplot as plt
import numpy as np
import os

path = "TestPhantoms/phantom_"

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
for i in range(1, 5):
    dir = path+str(i)
    phantom_name = "0.npy"
    phantom_path = os.path.join(dir,phantom_name)
    phantom = np.load(phantom_path)
    ax[i-1].imshow(phantom, cmap='gray')
    ax[i-1].xaxis.set_visible(False)
    ax[i-1].yaxis.set_visible(False)
    ax[i-1].set_title(f"Phantom ({i})", fontsize=22)
plt.subplots_adjust(bottom=0.1,wspace=0.01,hspace=0.4)
plt.savefig("phantoms")