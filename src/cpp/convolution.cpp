#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <omp.h>  // OpenMP for multi-threading
#include <algorithm>  // For std::min and std::max
#include <iostream>
#include <memory>
#include <cmath>  // For ceil
#include "convolution.h"

// Utility function (just as an example, replace with your actual logic)
uint64_t nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec*1000000000 + ts.tv_nsec;
}

// Naive OpenMP convolution
void naive_omp_convolution(const unsigned char* image, const float* ker, unsigned char* out, const int H, const int W) {
    int ker_r = KER / 2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);
    
    #pragma omp parallel for num_threads(thread_num) shared(image, ker, out, ker_r) schedule(static, chunk)
    for (int i = ker_r; i < W - ker_r; i++) {
        for (int j = ker_r; j < H - ker_r; j++) {
            float temp = 0;
            for (int k = 0; k < KER; k++) {
                for (int l = 0; l < KER; l++) {
                    temp += ker[k * KER + l] * image[(j + k - ker_r) * W + (i + l - ker_r)];
                }
            }
            out[j * W + i] = static_cast<unsigned char>(std::min(std::max(temp, 0.0f), 255.0f));
        }
    }
}

// Vectorized OpenMP convolution
void vec_omp_convolution(const unsigned char* image, const float* ker, unsigned char* out, const int H, const int W) {
    int ker_r = KER / 2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);
    
    #pragma omp parallel for num_threads(thread_num) shared(image, ker, out, ker_r) schedule(static, chunk)
    for (int i = ker_r; i < H - ker_r; i++) {
        for (int j = ker_r; j < W - ker_r; j++) {
            float temp = 0;
            #pragma omp simd
            for (int k = 0; k < KER; k++) {
                for (int l = 0; l < KER; l++) {
                    temp += ker[k * KER + l] * image[(j + k - ker_r) * W + (i + l - ker_r)];
                }
            }
            out[j * W + i] = static_cast<unsigned char>(std::min(std::max(temp, 0.0f), 255.0f));
        }
    }
}

// Smart OpenMP convolution
void smart_omp_convolution(const unsigned char* image, const float* ker, unsigned char* out, const int H, const int W) {
    int ker_r = KER / 2;
    int thread_num = omp_get_max_threads();
    int chunk = ceil(H / (float) thread_num);

    #pragma omp parallel num_threads(thread_num) shared(image, ker, out, ker_r, chunk)
    {
        int start_row = omp_get_thread_num() * chunk;
        int end_row = (omp_get_thread_num() == thread_num - 1) ? H : start_row + chunk;
        
        std::unique_ptr<unsigned char[]> private_out = std::make_unique<unsigned char[]>(chunk * W);

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < KER; k++) {
                    for (int l = 0; l < KER; l++) {
                        if (i >= ker_r && i < H - ker_r && j >= ker_r && j < W - ker_r) {
                            private_out[(i - start_row) * W + j] += (unsigned char) image[(i - ker_r + k) * W + (j - ker_r + l)] * ker[k * KER + l];
                        } else {
                            private_out[(i - start_row) * W + j] = image[i * W + j];
                        }
                    }
                }
            }
        }

        // Merge private output into the main output matrix
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < W; j++) {
                out[i * W + j] = private_out[(i - start_row) * W + j];
            }
        }
    }
}


// Wrapper for OpenMP convolution. Selects the convolution type based on user input.
cv::Mat convolution_dispatcher(const cv::Mat &image, const float kernel_h[KER * KER], ConvolutionType conv_type) {
    cv::Mat out(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
    double start, end;

    // Select convolution type based on user input
    switch (conv_type) {
        case NAIVE:
            start = omp_get_wtime();
            naive_omp_convolution(image.data, kernel_h, out.data, image.rows, image.cols);
            end = omp_get_wtime();
            std::cout << "Naive OpenMP Convolution Time: " << (end - start) << " seconds" << std::endl;
            break;
        
        case VECTORIZED:
            start = omp_get_wtime();
            vec_omp_convolution(image.data, kernel_h, out.data, image.rows, image.cols);
            end = omp_get_wtime();
            std::cout << "Vectorized OpenMP Convolution Time: " << (end - start) << " seconds" << std::endl;
            break;
        
        case SMART:
            start = omp_get_wtime();
            smart_omp_convolution(image.data, kernel_h, out.data, image.rows, image.cols);
            end = omp_get_wtime();
            std::cout << "Smart OpenMP Convolution Time: " << (end - start) << " seconds" << std::endl;
            break;
        
        default:
            std::cerr << "Invalid Convolution Type!" << std::endl;
            break;
    }

    return out;
}

// Python binding for the convolution function
static PyObject* conv2d(PyObject* self, PyObject* args) {
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

    // Create an output array for the result
    npy_intp output_dims[2] = {image_height - kernel_height + 1, image_width - kernel_width + 1};
    PyObject* output_array = PyArray_SimpleNew(2, output_dims, NPY_DOUBLE);
    double* output = (double*)PyArray_DATA(output_array);

    // Perform the convolution (multi-threaded with OpenMP)
    #pragma omp parallel for collapse(2)  // Parallelize the loops over image dimensions
    for (int i = 0; i < image_height - kernel_height + 1; i++) {
        for (int j = 0; j < image_width - kernel_width + 1; j++) {
            double sum = 0.0;
            #pragma omp simd  // SIMD optimization for the inner loop
            for (int ki = 0; ki < kernel_height; ki++) {
                for (int kj = 0; kj < kernel_width; kj++) {
                    sum += image[(i + ki) * image_width + (j + kj)] * kernel[ki * kernel_width + kj];
                }
            }
            output[i * (image_width - kernel_width + 1) + j] = sum;
        }
    }

    // Return the result as a numpy array
    return output_array;
}

// Method table for the module
static PyMethodDef ConvolveMethods[] = {
    {"convolve", conv2d, METH_VARARGS, "Perform 2D convolution."},
    {NULL, NULL, 0, NULL}  // Sentinel value
};

// Module definition
static struct PyModuleDef convolve_module = {
    PyModuleDef_HEAD_INIT,
    "conv2d",  // Module name
    "2D Convolution in C with OpenMP",  // Module docstring
    -1,  // Keeps state in global variables
    ConvolveMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_convolve(void) {
    import_array();  // Initialize numpy
    return PyModule_Create(&convolve_module);
}
