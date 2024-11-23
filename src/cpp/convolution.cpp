#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <memory>
#include <cstring>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define TILE_SIZE 32
#define ALIGN_TO 16
#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline void* aligned_malloc(size_t size) {
    void* ptr = nullptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(size, ALIGN_TO);
    #else
        if (posix_memalign(&ptr, ALIGN_TO, size) != 0) ptr = nullptr;
    #endif
    return ptr;
}

inline void aligned_free(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

inline void fast_memcpy(unsigned char* dst, const unsigned char* src, size_t size) {
    size_t blocks = size / sizeof(uint64_t);
    size_t remain = size % sizeof(uint64_t);
    
    uint64_t* dst64 = (uint64_t*)dst;
    const uint64_t* src64 = (const uint64_t*)src;
    
    while (blocks--) *dst64++ = *src64++;
    
    dst = (unsigned char*)dst64;
    src = (const unsigned char*)src64;
    while (remain--) *dst++ = *src++;
}

void fast_small_kernel_conv(const unsigned char* __restrict__ input,
                            unsigned char* __restrict__ output,
                            const float* __restrict__ kernel,
                            const int height,
                            const int width,
                            const int kernel_size) {
    const int kernel_radius = kernel_size / 2;
    const int row_stride = width;
    const float* kernel_ptr = kernel;
    
    float kernel_sum = 0.0f;
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernel_sum += kernel[i];
    }
    const float scale = (kernel_sum != 0.0f) ? 1.0f / kernel_sum : 1.0f;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int tile_y = kernel_radius; tile_y < height - kernel_radius; tile_y += TILE_SIZE) {
        for (int tile_x = kernel_radius; tile_x < width - kernel_radius; tile_x += TILE_SIZE) {
            const int tile_h = MIN(TILE_SIZE, height - kernel_radius - tile_y);
            const int tile_w = MIN(TILE_SIZE, width - kernel_radius - tile_x);
            
            for (int y = 0; y < tile_h; ++y) {
                const int global_y = tile_y + y;
                unsigned char* out_row = output + global_y * row_stride + tile_x;
                
                for (int x = 0; x < tile_w; ++x) {
                    float sum = 0.0f;
                    const unsigned char* in_ptr = input + (global_y - kernel_radius) * row_stride + 
                                                (tile_x + x - kernel_radius);
                    
                    if (kernel_size == 3) {
                        sum += in_ptr[0] * kernel_ptr[0];
                        sum += in_ptr[1] * kernel_ptr[1];
                        sum += in_ptr[2] * kernel_ptr[2];
                        in_ptr += row_stride;
                        sum += in_ptr[0] * kernel_ptr[3];
                        sum += in_ptr[1] * kernel_ptr[4];
                        sum += in_ptr[2] * kernel_ptr[5];
                        in_ptr += row_stride;
                        sum += in_ptr[0] * kernel_ptr[6];
                        sum += in_ptr[1] * kernel_ptr[7];
                        sum += in_ptr[2] * kernel_ptr[8];
                    } else {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            const unsigned char* in_kernel_row = in_ptr + ky * row_stride;
                            const float* kernel_row = kernel_ptr + ky * kernel_size;
                            
                            #pragma omp simd reduction(+:sum)
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                sum += in_kernel_row[kx] * kernel_row[kx];
                            }
                        }
                    }
                    
                    sum *= scale;
                    out_row[x] = (unsigned char)(sum < 0 ? 0 : (sum > 255 ? 255 : sum));
                }
            }
        }
    }
}

void optimized_convolution(const unsigned char* input,
                           unsigned char* output,
                           const float* kernel,
                           const int height,
                           const int width,
                           const int kernel_size) {
    const int kernel_radius = kernel_size / 2;
    
    std::memset(output, 0, height * width);
    fast_small_kernel_conv(input, output, kernel, height, width, kernel_size);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < kernel_radius; ++x) {
            output[y * width + x] = input[y * width + kernel_radius];
            output[y * width + (width - 1 - x)] = input[y * width + (width - 1 - kernel_radius)];
        }
    }
    
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < kernel_radius; ++y) {
            output[y * width + x] = output[kernel_radius * width + x];
            output[(height - 1 - y) * width + x] = output[(height - 1 - kernel_radius) * width + x];
        }
    }
}

static PyObject* conv2d(PyObject* self, PyObject* args) {
    PyObject *image_array, *kernel_array;
    
    if (!PyArg_ParseTuple(args, "OO", &image_array, &kernel_array)) {
        return NULL;
    }
    
    if (!PyArray_Check(image_array) || !PyArray_Check(kernel_array)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be numpy arrays");
        return NULL;
    }
    
    const int height = PyArray_DIM((PyArrayObject*)image_array, 0);
    const int width = PyArray_DIM((PyArrayObject*)image_array, 1);
    const int kernel_size = PyArray_DIM((PyArrayObject*)kernel_array, 0);
    
    npy_intp dims[2] = {height, width};
    PyObject* output_array = PyArray_SimpleNew(2, dims, NPY_UINT8);
    
    float* kernel = (float*)aligned_malloc(kernel_size * kernel_size * sizeof(float));
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        kernel[i] = *(float*)PyArray_GETPTR1((PyArrayObject*)kernel_array, i);
    }
    
    optimized_convolution(
        (unsigned char*)PyArray_DATA((PyArrayObject*)image_array),
        (unsigned char*)PyArray_DATA((PyArrayObject*)output_array),
        kernel,
        height,
        width,
        kernel_size
    );
    
    aligned_free(kernel);
    return output_array;
}

static PyMethodDef ConvolveMethods[] = {
    {"convolve", conv2d, METH_VARARGS, "Optimized 2D convolution"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef convolve_module = {
    PyModuleDef_HEAD_INIT,
    "conv2d",
    "High-performance 2D Convolution",
    -1,
    ConvolveMethods
};

PyMODINIT_FUNC PyInit_convolve(void) {
    import_array();
    return PyModule_Create(&convolve_module);
}
