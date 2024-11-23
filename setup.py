from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import shutil
import platform
import numpy as np
import pybind11

# Determine platform-specific settings
if platform.system() == "Windows":
    lib_ext = ".dll"
    extra_compile_args = ['/std:c++17', '/openmp', '/O2']
    extra_link_args = []
elif platform.system() == "Darwin":
    lib_ext = ".dylib"
    extra_compile_args = ['-std=c++17', '-O3', '-march=native', '-ffast-math']
    extra_link_args = ['-shared']
else:  # Linux
    lib_ext = ".so"
    extra_compile_args = ['-std=c++17', '-fopenmp', '-O3', '-march=native', '-ffast-math']
    extra_link_args = ['-fopenmp', '-shared']

# Define the shared library extension
matrix_multiply_lib = Extension(
    'torchbits.mx',  # Changed to include package name
    sources=['src/cpp/matrix_multiply.cpp'],
    include_dirs=[
        np.get_include(),
        pybind11.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++'
)

# Define the PyBind11 module extension
conv2d_ext = Extension(  # Renamed for clarity
    'torchbits.conv2d',
    sources=["src/cpp/convolution.cpp"],
    include_dirs=[
        np.get_include(),
        pybind11.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++'
)

# class CustomBuildExt(build_ext):
#     """Custom build_ext command to handle both extensions"""
    
#     def run(self):
#         # Build the PyBind11 module extension
#         build_ext.run(self)
        
#         # Build the shared library
#         self.build_shared_lib()
        
#         # Create lib directory if it doesn't exist
#         lib_dir = os.path.join(self.build_lib,'lib')
#         os.makedirs(lib_dir, exist_ok=True)
        
#         # Copy the shared library to the package
#         shared_lib_name = f"libmatrix_multiply{lib_ext}"
#         src_path = os.path.join(self.build_temp, shared_lib_name)
#         dst_path = os.path.join(lib_dir, shared_lib_name)
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dst_path)
#         else:
#             print(f"Warning: Source library {src_path} not found")

#     def build_shared_lib(self):
#         """Build the shared library"""
#         try:
#             # Create build directory if it doesn't exist
#             os.makedirs(self.build_temp, exist_ok=True)
            
#             # Compile the shared library
#             self.compiler.compile(
#                 sources=['src/cpp/matrix_multiply.cpp'],
#                 output_dir=self.build_temp,
#                 include_dirs=[
#                     np.get_include(),
#                     pybind11.get_include(),
#                 ],
#             )
            
#             # Link the shared library
#             objects = [os.path.join(self.build_temp, 'matrix_multiply.o')]
#             output_lib = os.path.join(self.build_temp, f"libmatrix_multiply{lib_ext}")
#             self.compiler.link_shared_object(
#                 objects,
#                 output_lib,
#                 extra_preargs=extra_link_args
#             )
#         except Exception as e:
#             print(f"Error building shared library: {e}")
#             raise

# Ensure the package structure exists
packages = ['torchbits.image', 'torchbits.lib']
for package in packages:
    os.makedirs(package.replace('.', '/'), exist_ok=True)

setup(
    name='torchbits',
    version='0.1.0',
    packages=packages,
    ext_modules=[conv2d_ext, matrix_multiply_lib],
    # cmdclass={'build_ext': CustomBuildExt},
    # package_data={
    #     'lib': [f'libmatrix_multiply{lib_ext}'],
    # },
    # include_package_data=True,
    install_requires=[
        'numpy',
        'pillow',
        'pybind11>=2.6.0',
    ],
    python_requires='>=3.6',
)