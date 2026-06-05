import matplotlib.pyplot as plt
import numpy as np


path = "TestPhantoms/phantom_3/6.npy"
phantom = np.load(path)

plt.imshow(phantom, cmap="gray")
plt.show()