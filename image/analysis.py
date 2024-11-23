import importlib
import convolution_module
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


def sobel_edge_detection(image: xp.ndarray,conv_type:int=1) -> xp.ndarray:
    Kx = xp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = xp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    if conv_type == 1:
        conv_type = convolution_module.ConvolutionType.NAIVE
    elif conv_type == 2:
        conv_type = convolution_module.ConvolutionType.VECTORIZED
    else: conv_type = convolution_module.ConvolutionType.SMART

    Gx = convolution_module.convolution_dispatcher(image, kx,conv_type)
    Gy = convolution_module.convolution_dispatcher(image, Ky,conv_type)
    return xp.sqrt(Gx**2 + Gy**2)


