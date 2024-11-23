import importlib
from typing import Tuple, Union, Optional
from PIL import Image, ImageOps
import numpy as np

# Check if GPU is available
try:
    import cupy as cp
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

# Use cupy if GPU is enabled, else use numpy
if gpu_enabled:
    xp = cp
else:
    xp = np


def histogram_equalization(image: xp.ndarray) -> xp.ndarray:
    hist, bins = xp.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf[-1]
    return np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)

