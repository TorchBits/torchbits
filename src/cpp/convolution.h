#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <Python.h>
#include <numpy/arrayobject.h>

// Function declarations
void optimized_convolution(const unsigned char* input,
                         unsigned char* output,
                         const float* kernel,
                         const int height,
                         const int width,
                         const int kernel_size);

#endif // CONVOLUTION_H