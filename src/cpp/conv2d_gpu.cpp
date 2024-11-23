#include <Python.h>
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>

// CUDA kernel for convolution
__global__ void convolve_kernel(double* d_image, double* d_kernel, double* d_output, int image_width, int image_height, int kernel_width, int kernel_height) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx < image_width - kernel_width + 1 && ty < image_height - kernel_height + 1) {
        double sum = 0.0;
        for (int ki = 0; ki < kernel_height; ki++) {
            for (int kj = 0; kj < kernel_width; kj++) {
                sum += d_image[(ty + ki) * image_width + (tx + kj)] * d_kernel[ki * kernel_width + kj];
            }
        }
        d_output[ty * (image_width - kernel_width + 1) + tx] = sum;
    }
}

// Wrapper for CUDA convolution
static PyObject* convolve_gpu(PyObject* self, PyObject* args) {
    PyArrayObject *image_array, *kernel_array;

    // Parse the input arguments (image and kernel)
    if (!PyArg_ParseTuple(args, "OO", &image_array, &kernel_array)) {
        return NULL;
    }

    // Ensure the image and kernel are numpy arrays
    if (!PyArray_Check(image_array) || !PyArray_Check(kernel_array)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be numpy arrays.");
        return NULL;
    }

    // Get the image and kernel as C arrays
    double* image = (double*)PyArray_DATA(image_array);
    double* kernel = (double*)PyArray_DATA(kernel_array);
    npy_intp* image_shape = PyArray_DIMS(image_array);
    npy_intp* kernel_shape = PyArray_DIMS(kernel_array);

    int image_width = image_shape[1];
    int image_height = image_shape[0];
    int kernel_width = kernel_shape[1];
    int kernel_height = kernel_shape[0];

    // Allocate memory for the output array
    npy_intp output_dims[2] = {image_height - kernel_height + 1, image_width - kernel_width + 1};
    PyObject* output_array = PyArray_SimpleNew(2, output_dims, NPY_DOUBLE);
    double* output = (double*)PyArray_DATA(output_array);

    // Allocate device memory for image, kernel, and output
    double *d_image, *d_kernel, *d_output;
    cudaMalloc((void**)&d_image, image_height * image_width * sizeof(double));
    cudaMalloc((void**)&d_kernel, kernel_height * kernel_width * sizeof(double));
    cudaMalloc((void**)&d_output, (image_height - kernel_height + 1) * (image_width - kernel_width + 1) * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_image, image, image_height * image_width * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_height * kernel_width * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 block_size(16, 16);
    dim3 grid_size((image_width - kernel_width + 1 + block_size.x - 1) / block_size.x, 
                   (image_height - kernel_height + 1 + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    convolve_kernel<<<grid_size, block_size>>>(d_image, d_kernel, d_output, image_width, image_height, kernel_width, kernel_height);

    // Copy the result back to host
    cudaMemcpy(output, d_output, (image_height - kernel_height + 1) * (image_width - kernel_width + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Return the result as a numpy array
    return output_array;
}

// Method table for the module
static PyMethodDef ConvolveMethods[] = {
    {"convolve_gpu", convolve_gpu, METH_VARARGS, "Perform 2D convolution using GPU (CUDA)."},
    {NULL, NULL, 0, NULL}  // Sentinel value
};

// Module definition
static struct PyModuleDef convolve_module = {
    PyModuleDef_HEAD_INIT,
    "convolve",  // Module name
    "2D Convolution with OpenMP and CUDA",  // Module docstring
    -1,  // Keeps state in global variables
    ConvolveMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_convolve(void) {
    import_array();  // Initialize numpy
    return PyModule_Create(&convolve_module);
}
