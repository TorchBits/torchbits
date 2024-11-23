// convolution.h

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <memory>

#define KER 3  // Define the kernel size (can be changed as needed)

enum ConvolutionType {
    NAIVE,
    VECTORIZED,
    SMART
};

// Function declarations
cv::Mat convolution_dispatcher(const cv::Mat &image, const float kernel_h[KER * KER], ConvolutionType conv_type);

void naive_omp_convolution(const uchar *image, const float *ker, uchar *out, const int H, const int W);
void vec_omp_convolution(const uchar *image, const float *ker, uchar *out, const int H, const int W);
void smart_omp_convolution(const uchar *image, const float *ker, uchar *out, const int H, const int W);

#endif  // CONVOLUTION_H
