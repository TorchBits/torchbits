// pybind_convolution.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "convolution.h" 
#include <opencv2/opencv.hpp>
 // Include header file instead of the cpp file

namespace py = pybind11;

// Binding for the convolution_dispatcher function
py::array_t<uint8_t> convolution_dispatcher_py(py::array_t<uint8_t> input, py::array_t<float> kernel, int conv_type) {
    // Convert input image to OpenCV format
    py::buffer_info input_info = input.request();
    int H = input_info.shape[0];
    int W = input_info.shape[1];
    cv::Mat image(H, W, CV_8UC1, static_cast<uint8_t*>(input_info.ptr));

    // Convert kernel to C++ array
    py::buffer_info kernel_info = kernel.request();
    if (kernel_info.size != KER * KER) {
        throw std::runtime_error("Kernel size mismatch. Expected " + std::to_string(KER * KER));
    }
    float kernel_data[KER * KER];
    std::memcpy(kernel_data, kernel_info.ptr, KER * KER * sizeof(float));

    // Run convolution and get the output image
    cv::Mat output_image = convolution_dispatcher(image, kernel_data, static_cast<ConvolutionType>(conv_type));

    // Convert the output image to a numpy array
    auto result = py::array_t<uint8_t>({H, W});
    std::memcpy(result.mutable_data(), output_image.data, H * W * sizeof(uint8_t));

    return result;
}

PYBIND11_MODULE(convolution_module, m) {
    m.doc() = "Python bindings for C++ convolution functions using OpenMP";

    // Expose the ConvolutionType enum to Python
    py::enum_<ConvolutionType>(m, "ConvolutionType")
        .value("NAIVE", ConvolutionType::NAIVE)
        .value("VECTORIZED", ConvolutionType::VECTORIZED)
        .value("SMART", ConvolutionType::SMART)
        .export_values();

    // Expose the convolution dispatcher function
    m.def("convolution_dispatcher", &convolution_dispatcher_py, "Convolution dispatcher function",
          py::arg("input"), py::arg("kernel"), py::arg("conv_type"));
}
