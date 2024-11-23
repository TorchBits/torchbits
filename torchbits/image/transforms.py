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


def _load_image(image: Union[str, xp.ndarray]) -> Image.Image:
    """Helper function to load an image from either a path or a numpy array."""
    if isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, xp.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    else:
        raise ValueError("Input must be a file path or a numpy array.")


def scale(image: xp.ndarray, new_size: Tuple[int, int]) -> xp.ndarray:
    """
    Resizes or scales the image to the specified size.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        new_size (tuple): The target size (width, height) for resizing.

    Returns:
        xp.ndarray: Resized image as a numpy/cupy array.
    """
    img = _load_image(image)
    scaled = xp.array(img.resize(new_size, Image.BICUBIC))
    return xp.array(scaled)


def rotate(image: xp.ndarray, angle: float, expand: bool = True) -> xp.ndarray:
    """
    Rotates the image by a specific angle.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        angle (float): Angle (in degrees) to rotate the image.
        expand (bool): Whether to expand the output image to fit the rotated image.

    Returns:
        xp.ndarray: Rotated image as a numpy/cupy array.
    """
    img = _load_image(image)
    rotated = img.rotate(angle, expand=expand)
    return xp.array(xp.array(rotated))


def contrast_stretching(image: xp.ndarray, min_val: int, max_val: int) -> xp.ndarray:
    """
    Applies contrast stretching on the image.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        min_val (int): Minimum intensity value for contrast stretching.
        max_val (int): Maximum intensity value for contrast stretching.

    Returns:
        xp.ndarray: Contrast stretched image as a numpy/cupy array.
    """
    min_image = xp.min(image)
    max_image = xp.max(image)
    stretched_image = min_val + (image - min_image) * (max_val - min_val) / (max_image - min_image)
    return xp.clip(stretched_image, min_val, max_val).astype(xp.uint8)


def log_transform(image: xp.ndarray, c: int) -> xp.ndarray:
    """
    Performs log transformation on the image.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        c (int): Constant multiplier for the log transformation.

    Returns:
        xp.ndarray: Log-transformed image as a numpy/cupy array.
    """
    return c * xp.log1p(image)


def gamma_transform(image: xp.ndarray, c: int, gamma: float, epsilon: int = 0) -> xp.ndarray:
    """
    Applies gamma correction on the image.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        c (int): Constant multiplier.
        gamma (float): Gamma value for correction.
        epsilon (int): Small value to prevent log(0) errors.

    Returns:
        xp.ndarray: Gamma-transformed image as a numpy/cupy array.
    """
    transformed_image = c * ((image + epsilon) ** gamma)
    return xp.clip(transformed_image, 0, 255).astype(xp.uint8)



def add_gaussian_noise(image: xp.ndarray, mean: float = 0, sigma: float = 25) -> xp.ndarray:
    """
    Adds Gaussian noise to an image.

    Args:
        image (xp.ndarray): Input image (NumPy or CuPy array).
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        xp.ndarray: Image with added Gaussian noise as a numpy/cupy array.
    """
    gaussian_noise = xp.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    return xp.clip(noisy_image, 0, 255).astype(xp.uint8)



def fft2(image: np.ndarray) -> np.ndarray:
    return xp.fft.fft2(image)

def ifft2(image: np.ndarray) -> np.ndarray:
    return xp.fft.ifft2(image)


def fftshift(image: np.ndarray) -> np.ndarray:
    return xp.fft.fftshift(image)

    
def threshold(image: np.ndarray, threshold_value: float) -> np.ndarray:
    return xp.where(image > threshold_value, 1, 0)
