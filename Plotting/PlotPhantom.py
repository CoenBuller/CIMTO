import matplotlib.pyplot as plt
from numpy.typing import NDArray

def PlotPhantom(img: NDArray) -> None:
    """Can plot a single phantom"""
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()