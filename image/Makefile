# # Compiler and flags
# CXX = g++
# CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -fopenmp

# # Pybind11 and Python3 configuration
# PYTHON_VERSION = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))")
# PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
# PYTHON_LIB = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

# # OpenCV configuration
# OPENCV_LIB = $(shell pkg-config --cflags --libs opencv4)

# # Output module name
# MODULE_NAME = convolution_module$(PYTHON_VERSION).so

# # Source files
# SOURCES = bindings.cpp convolution.cpp utils.cpp

# # Compile target
# $(MODULE_NAME): $(SOURCES)
# 	$(CXX) $(CXXFLAGS) $(SOURCES) -I$(PYTHON_INCLUDE) -I/usr/include/opencv4 \
# 		`python3 -m pybind11 --includes` $(OPENCV_LIB) -o $(MODULE_NAME)

# # Clean command
# clean:
# 	rm -f $(MODULE_NAME)


# Define variables
PYTHON = python3
BUILD_DIR = build  # Path to your build directory
SO_FILE = $(BUILD_DIR)/convolution_module.so  # The .so file name
INSTALL_DIR = $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_lib())")

# Install target to copy the .so file to the Python site-packages directory
install:
	@echo "Installing $(SO_FILE) to $(INSTALL_DIR)..."
	cp $(SO_FILE) $(INSTALL_DIR)/

# Clean target to remove the .so file
clean:
	rm -f $(SO_FILE)

.PHONY: install clean
