#pragma once

#include <time.h>
#include <omp.h>
#include <cstdint>
#include <cmath>
#define KER 7

void gaussian_kernel (float* kernel, float sigma) {

    float sum = 0.0;
    int center = KER / 2;

    for (int i = 0; i < KER; ++i) {
        for (int j = 0; j < KER; ++j) {
            int x = i - center;
            int y = j - center;
            kernel[i * KER + j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * KER + j];
        }
    }

    for (int i = 0; i < KER; ++i) {
        for (int j = 0; j < KER; ++j) {
            kernel[i * KER + j] /= sum;
        }
    }
}

void emboss_kernel (float* kernel) {
    kernel[0] = -2.0;
    kernel[1] = -1.0;
    kernel[2] = 0.0;
    kernel[3] = -1.0;
    kernel[4] = 1.0;
    kernel[5] = 1.0;
    kernel[6] = 0.0;
    kernel[7] = 1.0;
    kernel[8] = 2.0;
}

void sharpen_kernel(float* kernel) {
    kernel[0] = 0.0;
    kernel[1] = -1.0;
    kernel[2] = 0.0;
    kernel[3] = -1.0;
    kernel[4] = 5.0;
    kernel[5] = -1.0;
    kernel[6] = 0.0;
    kernel[7] = -1.0;
    kernel[8] = 0.0;
}

void edge_kernel(float* kernel) {
    kernel[0] = 0.0;
    kernel[1] = 1.0;
    kernel[2] = 0.0;
    kernel[3] = 1.0;
    kernel[4] = -4.0;
    kernel[5] = 1.0;
    kernel[6] = 0.0;
    kernel[7] = 1.0;
    kernel[8] = 0.0;
}
