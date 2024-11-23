from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import shutil
import platform
import numpy as np

# Determine the file extension for shared libraries based on the platform
if platform.system() == "Windows":
    lib_ext = ".dll"
elif platform.system() == "Darwin":
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

# Define the shared library extension
matrix_multiply_lib = Extension(
    'matrix_multiply',
    sources=['src/cpp/matrix_multiply.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-std=c++11', '-fPIC'],
    extra_link_args=['-shared'],
)

# Define the Python module extension
module1_ext = Extension(
    'torchbits.convolution',
    sources=["src/cpp/bindings.cpp", "src/cpp/convolution.cpp","src/cpp/utils.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=['-std=c++11'],
)

class CustomBuildExt(build_ext):
    """Custom build_ext command to handle both extensions"""
    
    def run(self):
        # Build the regular Python extension module
        build_ext.run(self)
        
        # Build the shared library
        self.build_shared_lib()
        
        # Create lib directory if it doesn't exist
        lib_dir = os.path.join(self.build_lib, 'your_package', 'lib')
        os.makedirs(lib_dir, exist_ok=True)
        
        # Copy the shared library to the package
        shared_lib_name = f"libmatrix_multiply{lib_ext}"
        src_path = os.path.join(self.build_temp, shared_lib_name)
        dst_path = os.path.join(lib_dir, shared_lib_name)
        shutil.copy2(src_path, dst_path)

    def build_shared_lib(self):
        """Build the shared library"""
        # Create build directory if it doesn't exist
        os.makedirs(self.build_temp, exist_ok=True)
        
        # Compile the shared library
        self.compiler.compile(
            sources=['src/cpp/matrix_multiply.cpp'],
            output_dir=self.build_temp,
            include_dirs=[np.get_include()],
            extra_preargs=['-std=c++11', '-fPIC']
        )
        
        # Link the shared library
        objects = [os.path.join(self.build_temp, 'matrix_multiply.o')]
        self.compiler.link_shared_lib(
            objects=objects,
            output_libname='matrix_multiply',
            output_dir=self.build_temp,
            extra_preargs=['-shared']
        )

setup(
    name='torchbits',
    version='0.1.0',
    packages=['torchbits', 'torchbits.lib'],
    ext_modules=[module1_ext],
    cmdclass={'build_ext': CustomBuildExt},
    package_data={
        'torchbits.lib': [f'libmatrix_multiply{lib_ext}'],
    },
    include_package_data=True,
    install_requires=['numpy','cupy','pillow'],
    python_requires='>=3.6',
)