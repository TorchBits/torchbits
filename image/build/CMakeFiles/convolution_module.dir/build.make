# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ace/Downloads/torchbit/torchbit/image

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ace/Downloads/torchbit/torchbit/image/build

# Include any dependencies generated for this target.
include CMakeFiles/convolution_module.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/convolution_module.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/convolution_module.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convolution_module.dir/flags.make

CMakeFiles/convolution_module.dir/bindings.cpp.o: CMakeFiles/convolution_module.dir/flags.make
CMakeFiles/convolution_module.dir/bindings.cpp.o: /home/ace/Downloads/torchbit/torchbit/image/bindings.cpp
CMakeFiles/convolution_module.dir/bindings.cpp.o: CMakeFiles/convolution_module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ace/Downloads/torchbit/torchbit/image/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/convolution_module.dir/bindings.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/convolution_module.dir/bindings.cpp.o -MF CMakeFiles/convolution_module.dir/bindings.cpp.o.d -o CMakeFiles/convolution_module.dir/bindings.cpp.o -c /home/ace/Downloads/torchbit/torchbit/image/bindings.cpp

CMakeFiles/convolution_module.dir/bindings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/convolution_module.dir/bindings.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ace/Downloads/torchbit/torchbit/image/bindings.cpp > CMakeFiles/convolution_module.dir/bindings.cpp.i

CMakeFiles/convolution_module.dir/bindings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/convolution_module.dir/bindings.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ace/Downloads/torchbit/torchbit/image/bindings.cpp -o CMakeFiles/convolution_module.dir/bindings.cpp.s

CMakeFiles/convolution_module.dir/convolution.cpp.o: CMakeFiles/convolution_module.dir/flags.make
CMakeFiles/convolution_module.dir/convolution.cpp.o: /home/ace/Downloads/torchbit/torchbit/image/convolution.cpp
CMakeFiles/convolution_module.dir/convolution.cpp.o: CMakeFiles/convolution_module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ace/Downloads/torchbit/torchbit/image/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/convolution_module.dir/convolution.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/convolution_module.dir/convolution.cpp.o -MF CMakeFiles/convolution_module.dir/convolution.cpp.o.d -o CMakeFiles/convolution_module.dir/convolution.cpp.o -c /home/ace/Downloads/torchbit/torchbit/image/convolution.cpp

CMakeFiles/convolution_module.dir/convolution.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/convolution_module.dir/convolution.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ace/Downloads/torchbit/torchbit/image/convolution.cpp > CMakeFiles/convolution_module.dir/convolution.cpp.i

CMakeFiles/convolution_module.dir/convolution.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/convolution_module.dir/convolution.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ace/Downloads/torchbit/torchbit/image/convolution.cpp -o CMakeFiles/convolution_module.dir/convolution.cpp.s

CMakeFiles/convolution_module.dir/utils.cpp.o: CMakeFiles/convolution_module.dir/flags.make
CMakeFiles/convolution_module.dir/utils.cpp.o: /home/ace/Downloads/torchbit/torchbit/image/utils.cpp
CMakeFiles/convolution_module.dir/utils.cpp.o: CMakeFiles/convolution_module.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ace/Downloads/torchbit/torchbit/image/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/convolution_module.dir/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/convolution_module.dir/utils.cpp.o -MF CMakeFiles/convolution_module.dir/utils.cpp.o.d -o CMakeFiles/convolution_module.dir/utils.cpp.o -c /home/ace/Downloads/torchbit/torchbit/image/utils.cpp

CMakeFiles/convolution_module.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/convolution_module.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ace/Downloads/torchbit/torchbit/image/utils.cpp > CMakeFiles/convolution_module.dir/utils.cpp.i

CMakeFiles/convolution_module.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/convolution_module.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ace/Downloads/torchbit/torchbit/image/utils.cpp -o CMakeFiles/convolution_module.dir/utils.cpp.s

# Object files for target convolution_module
convolution_module_OBJECTS = \
"CMakeFiles/convolution_module.dir/bindings.cpp.o" \
"CMakeFiles/convolution_module.dir/convolution.cpp.o" \
"CMakeFiles/convolution_module.dir/utils.cpp.o"

# External object files for target convolution_module
convolution_module_EXTERNAL_OBJECTS =

convolution_module.so: CMakeFiles/convolution_module.dir/bindings.cpp.o
convolution_module.so: CMakeFiles/convolution_module.dir/convolution.cpp.o
convolution_module.so: CMakeFiles/convolution_module.dir/utils.cpp.o
convolution_module.so: CMakeFiles/convolution_module.dir/build.make
convolution_module.so: /usr/lib64/libopencv_imgproc.so.4.8.1
convolution_module.so: /usr/lib64/libopencv_core.so.4.8.1
convolution_module.so: /usr/lib/gcc/x86_64-redhat-linux/13/libgomp.so
convolution_module.so: /usr/lib64/libpthread.a
convolution_module.so: CMakeFiles/convolution_module.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/ace/Downloads/torchbit/torchbit/image/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared module convolution_module.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convolution_module.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convolution_module.dir/build: convolution_module.so
.PHONY : CMakeFiles/convolution_module.dir/build

CMakeFiles/convolution_module.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convolution_module.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convolution_module.dir/clean

CMakeFiles/convolution_module.dir/depend:
	cd /home/ace/Downloads/torchbit/torchbit/image/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ace/Downloads/torchbit/torchbit/image /home/ace/Downloads/torchbit/torchbit/image /home/ace/Downloads/torchbit/torchbit/image/build /home/ace/Downloads/torchbit/torchbit/image/build /home/ace/Downloads/torchbit/torchbit/image/build/CMakeFiles/convolution_module.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/convolution_module.dir/depend
