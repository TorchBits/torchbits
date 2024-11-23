#__init__.py file
from .image import *  # Import the Python module extension
from .lib import matrix_multiply_lib  # Import the shared library
from .conv2d import convolve
import numpy as np

def apply_convolution(image, kernel):
    """
    Apply convolution to an image with the given kernel.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (grayscale, uint8)
    kernel : numpy.ndarray
        Convolution kernel (float32)
        
    Returns:
    --------
    numpy.ndarray
        Convolved image
    """
    # Ensure correct input types
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if kernel.dtype != np.float32:
        kernel = kernel.astype(np.float32)
    
    # Ensure kernel is square
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be square")
    
    return convolve(image, kernel)

# Common kernel definitions
class Kernels:
    @staticmethod
    def gaussian(size=3, sigma=1.0):
        """Generate a Gaussian kernel"""
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return (kernel / np.sum(kernel)).astype(np.float32)
    
    @staticmethod
    def sobel_x():
        """Sobel X-direction kernel"""
        return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    @staticmethod
    def sobel_y():
        """Sobel Y-direction kernel"""
        return np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
__version__ = '0.1.0'