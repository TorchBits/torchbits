from setuptools import setup, find_packages

setup(
    name="torchbit",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x"]  # Replace with the appropriate version tag for CuPy
    },
)
