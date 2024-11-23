import ctypes
import importlib
import numpy as np
import torchbits.mx

try:
    import cupy as cp 
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

if gpu_enabled:
    import cupy as xp 
else:
    import numpy as xp

class Linear:
    def __init__(self, in_features, out_features, use_gpu=False):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.weights = self.xp.random.randn(in_features, out_features).astype(self.xp.float32) * 0.01
        self.bias = self.xp.zeros(out_features, dtype=self.xp.float32)

    def forward(self, x):
        self.input = x
        if self.use_gpu:
            return cp.dot(x, self.weights) + self.bias
        else:
            torchbits.mx.matrix_multiply(
                x,self.weights
            ) + self.bias

    def backward(self, grad_output, learning_rate=0.01):
        grad_input = self.xp.dot(grad_output, self.weights.T)
        grad_weights = self.xp.dot(self.input.T, grad_output)
        grad_bias = self.xp.sum(grad_output, axis=0)

        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input

    def save(self, path_prefix):
        """Save weights and bias to .npy files."""
        np.save(f"{path_prefix}_weights.npy", self.weights.get() if self.use_gpu else self.weights)
        np.save(f"{path_prefix}_bias.npy", self.bias.get() if self.use_gpu else self.bias)

    def load(self, path_prefix):
        """Load weights and bias from .npy files."""
        self.weights = self.xp.array(np.load(f"{path_prefix}_weights.npy"), dtype=self.xp.float32)
        self.bias = self.xp.array(np.load(f"{path_prefix}_bias.npy"), dtype=self.xp.float32)

class MSELoss:
    def __call__(self, prediction, target):
        return ((prediction - target) ** 2).mean()

    def gradient(self, prediction, target):
        return 2 * (prediction - target) / target.size
