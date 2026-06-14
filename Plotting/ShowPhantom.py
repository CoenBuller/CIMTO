import matplotlib.pyplot as plt
import numpy as np
import argparse


""""Can be used to plot a specific phantom"""

# Pass phantom type and instance as argument when calling the python file (e.g. python -m Plotting.ShowPhantom -type 2 -instance 8)
parser = argparse.ArgumentParser()
parser.add_argument('-type', choices=['1', '2', '3', '4'], default='0')
parser.add_argument('-instance', choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] , default='0')
args = parser.parse_args()

path = f"TestPhantoms/phantom_{args.type}/{args.instance}.npy"
phantom = np.load(path)

plt.imshow(phantom, cmap="gray")
plt.savefig("phantomtest")