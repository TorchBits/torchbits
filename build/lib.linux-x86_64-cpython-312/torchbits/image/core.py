from PIL import Image
import importlib
try:
    import cupy as cp 
    gpu_enabled = True

except ImportError:
    gpu_enabled = False

if gpu_enabled:
    import cupy as xp 
else:
    import numpy as xp


class ExtendedImage(Image.Image):
    def __init__(self, image: Image.Image):
        self._image = image
        self._data = xp.array(image)

    def to_array(self):
        """Convert the image to a numpy array."""
        return self._data

    def from_array(self, arr: xp.ndarray):
        """Convert a numpy array back to an image."""
        self._data = arr
        self._image = Image.fromarray(arr)
        return self._image

    def apply_filter(self, kernel: xp.ndarray,filter):
        """Apply a custom filter to the image (e.g., convolution)."""
        # Assuming grayscale image for simplicity
        filtered_data = filter(self._data, kernel, mode='same', boundary='wrap')
        return self.from_array(filtered_data)

    def display(self):
        """Display the image."""
        self._image.show()
