from torchbit.image.core import ExtendedImage
import numpy as np

class BitTensor(ExtendedImage):
    def __init__(self, image):
        super().__init__(image)

    def apply_noise(self, noise_level: float,noise_type=gauss):
        """Apply Gaussian noise to the image."""
        noise = np.random.normal(0, noise_level, self._data.shape)
        noisy_image = self._data + noise
        noisy_image = np.clip(noisy_image, 0, 255)  # Clipping to valid image range
        return self.from_array(noisy_image)

    def fft_transform(self):
        """Apply FFT to the image (frequency domain analysis)."""
        fft_result = np.fft.fft2(self._data)
        fft_result_shifted = np.fft.fftshift(fft_result)  # Center the low frequencies
        return fft_result_shifted

    def apply_threshold(self, threshold: float):
        """Apply a thresholding operation."""
        thresholded_image = np.where(self._data > threshold, 255, 0)
        return self.from_array(thresholded_image)
