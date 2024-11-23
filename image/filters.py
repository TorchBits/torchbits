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

###################################### simple statistics and math operations######################################
def mean(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the mean of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to average
    Return: xp.array, the averaged image
    """
    return xp.mean(image, axis=axis)


def median(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the median of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to perform the function
    Return: xp.array, the median of the image
    """
    return xp.median(image, axis=axis)

def variance(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the variance of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to perform the function
    Return: xp.array, the variance of the image
    """
    return xp.var(image, axis=axis)

def std_deviation(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the standard deviation of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to perform the function
    Return: xp.array, the std of the image
    """
    return xp.std(image, axis=axis)

def max_value(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the max of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to perform the function
    Return: xp.array, the max value of image along a particular axis
    """
    return xp.max(image, axis=axis)

def min_value(image: xp.ndarray, axis: Optional[tuple] = None) -> xp.ndarray:
    """
    Calculates the min value of the image intensities along a particular axis
    Parameters:
    image: the input image
    axis: the axis to perform the function
    Return: xp.array, the min_value image
    """
    return xp.min(image, axis=axis)

def add(image: xp.ndarray, value: float) -> xp.ndarray:
    """
    Adds a value to the intensities of the image
    Parameters:
    image: the input image
    Return: xp.array, the added image
    """
    return xp.add(image, value)

def subtract(image: xp.ndarray, value: float) -> xp.ndarray:
    """
    Subtracts a value from the intensities of the image
    Parameters:
    image: the input image
    Return: xp.array, the subtrated image
    """
    return xp.subtract(image, value)

def multiply(image: xp.ndarray, value: float) -> xp.ndarray:
    """
    Multiplies the intensities of the image by a value
    Parameters:
    image: the input image
    Return: xp.array, the multiplied image
    """
    return xp.multiply(image, value)

def divide(image: xp.ndarray, value: float) -> xp.ndarray:
    """
    Divides the intensities of the image by a value
    Parameters:
    image: the xp.ndarray image
    value: the value to divide the intensities with
    Return: xp.ndarray, the divided image
    """
    return xp.divide(image, value)

def power(image: xp.ndarray, exponent: float) -> xp.ndarray:
    """ 
    Raises the image intensities by a certain power
    Paramters: 
    image: the xp.array image
    exponent: the power to be raised to
    return: the exponentiated image
    """
    return xp.power(image, exponent)


def translate(image: xp.ndarray, shift: tuple) -> xp.ndarray:
    """
    Function to shift image along a certain axis
    Parameters:
    image: the xp.ndarray image
    shift: a tuple of size (a,b)
    return : the translated image
    """
    return xp.roll(image, shift, axis=(0, 1))  # Shift along axes

####################################### filter functions#########################################

def median_filter(image: xp.ndarray, kernel_size: int = 3) -> xp.ndarray:
    padded_image = xp.pad(image, kernel_size // 2, mode='edge')
    result = xp.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = xp.median(neighborhood)
    return result

def gaussian_blur(image: xp.ndarray) -> xp.ndarray:
    kernel = xp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return convolve(image, kernel)


def box_filter(image: xp.ndarray,kernel_size:int) -> xp.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create the box filter kernel
    box_kernel = xp.ones((kernel_size, kernel_size), dtype=image.dtype)
    h,w = image.shape[0],image.shape[1]
    norm_const = 1/(h * w)
    box_kernel = norm_const * box_kernel
    return convolve(image,box_kernel)


def laplacian_filter(image: xp.ndarray) -> xp.ndarray:
    kernel = xp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve(image, kernel)


def custom_filter(image: xp.ndarray,kernel: xp.ndarray) -> xp.ndarray:
    return convolve(image,kernel)


def pad_image(image, pad_size):
    """Pad image with reflected borders"""
    return xp.pad(image, pad_size, mode='reflect')

def contraharmonic_mean_filter(image, window_size=3, Q=1.5):
    """
    Contraharmonic mean filter implementation
    Q > 0: Better at removing pepper noise
    Q < 0: Better at removing salt noise
    """
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            numerator = xp.sum(np.power(window, Q + 1))
            denominator = xp.sum(np.power(window, Q))
            if denominator != 0:
                result[i, j] = numerator / denominator
            else:
                result[i, j] = image[i, j]
    
    return result

def harmonic_mean_filter(image, window_size=3):
    """
    Harmonic mean filter implementation
    Good for salt noise and Gaussian noise
    """
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            denominator = xp.sum(1.0 / (window + xp.finfo(float).eps))
            if denominator != 0:
                result[i, j] = window_size * window_size / denominator
            else:
                result[i, j] = image[i, j]
    
    return result

def geometric_mean_filter(image, window_size=3):
    """
    Geometric mean filter implementation
    Good for Gaussian noise
    """
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            result[i, j] = xp.exp(xp.mean(xp.log(window + xp.finfo(float).eps)))
    
    return result

def alpha_trimmed_mean_filter(image, window_size=3, alpha=2):
    """
    Alpha-trimmed mean filter implementation
    Good for mixed noise patterns
    alpha: number of pixels to trim from each end
    """
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            flat_window = xp.sort(window.flatten())
            trimmed = flat_window[alpha:-alpha] if len(flat_window) > 2*alpha else flat_window
            result[i, j] = xp.mean(trimmed)
    
    return result

def adaptive_median_filter(image, max_window_size=7):
    """
    Adaptive median filter implementation
    Adapts window size based on local statistics
    """
    pad_size = max_window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window_size = 3
            while window_size <= max_window_size:
                half = window_size // 2
                window = padded[i+pad_size-half:i+pad_size+half+1,
                              j+pad_size-half:j+pad_size+half+1]
                
                z_min = xp.min(window)
                z_max = xp.max(window)
                z_med = xp.median(window)
                z_xy = padded[i+pad_size, j+pad_size]
                
                A1 = z_med - z_min
                A2 = z_med - z_max
                
                if A1 > 0 and A2 < 0:
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max
                    if B1 > 0 and B2 < 0:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    break
                else:
                    window_size += 2
                    if window_size > max_window_size:
                        result[i, j] = z_med
                        break
                        
    return result

def local_noise_reduction_filter(image, window_size=3, noise_variance=None):
    """
    Local noise reduction filter implementation
    Adapts to local image statistics
    """
    if noise_variance is None:
        # Estimate noise variance using median absolute deviation
        noise_variance = xp.median(xp.abs(image - xp.median(image))) / 0.6745
    
    pad_size = window_size // 2
    padded = pad_image(image, pad_size)
    result = xp.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            local_mean = xp.mean(window)
            local_variance = xp.var(window)
            
            if local_variance > 0:
                k = noise_variance / local_variance
                result[i, j] = local_mean + (1 - k) * (image[i, j] - local_mean)
            else:
                result[i, j] = local_mean
                
    return result
