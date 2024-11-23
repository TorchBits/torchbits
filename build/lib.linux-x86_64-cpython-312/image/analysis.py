import importlib
from .conv2d import convolve
from typing import Tuple, Union, Optional

try:
    import cupy as cp 
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

if gpu_enabled:
    import cupy as xp 
else:
    import numpy as xp


def sobel_edge_detection(image: xp.ndarray) -> xp.ndarray:
    Kx = xp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = xp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = convolve(image, kx)
    Gy = convolve(image, Ky)
    return xp.sqrt(Gx**2 + Gy**2)


