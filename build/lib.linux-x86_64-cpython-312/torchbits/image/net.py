import ctypes
import importlib
from typing import Tuple, Union, Optional
from torchbits.mx import matrix_multiply
try:
    import cupy as cp 
    gpu_enabled = True
except ImportError:
    gpu_enabled = False

if gpu_enabled:
    import cupy as xp 
else:
    import numpy as xp


# Define the matrix multiplication function in Python
def matrix_multiply(A, B):
    M, N = A.shape
    N, K = B.shape
    C = xp.zeros((M, K), dtype=np.float32)
    
    # Call the C++ function
    matrix_multiply(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
    )
    return C


class LinearLayer:
    def __init__(self, in_features, out_features, use_gpu=False):
        self.use_gpu = use_gpu
        xp = cp if use_gpu else np
        self.weights = xp.random.randn(in_features, out_features).astype(xp.float32) * 0.01
        self.bias = xp.zeros(out_features, dtype=xp.float32)

    def forward(self, x):
        self.input = x
        xp = cp if self.use_gpu else np
        if self.use_gpu:
            return cp.dot(x, self.weights) + self.bias
        else:
            return matrix_multiply(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate=0.01):
        xp = cp if self.use_gpu else np
        grad_input = xp.dot(grad_output, self.weights.T)
        grad_weights = xp.dot(self.input.T, grad_output)
        grad_bias = xp.sum(grad_output, axis=0)

        # Update parameters
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input



class MSELoss:
    def __call__(self, prediction, target):
        return ((prediction - target) ** 2).mean()

    def gradient(self, prediction, target):
        return 2 * (prediction - target) / target.size
