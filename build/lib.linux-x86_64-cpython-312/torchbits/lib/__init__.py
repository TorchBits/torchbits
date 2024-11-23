import os
import ctypes
import platform
from pathlib import Path

def get_lib_extension():
    if platform.system() == "Windows":
        return ".dll"
    elif platform.system() == "Darwin":
        return ".dylib"
    return ".so"

def load_matrix_multiply():
    # Get the directory where the shared library is stored
    lib_dir = Path(__file__).parent
    lib_name = f"libmatrix_multiply{get_lib_extension()}"
    lib_path = str(lib_dir / lib_name)
    
    try:
        return ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library: {e}")

# Export the library
matrix_multiply_lib = load_matrix_multiply()