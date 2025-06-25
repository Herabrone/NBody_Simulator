cmake_minimum_required(VERSION 3.18)
project(ForceGPU LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)

add_library(force_gpu MODULE
    force_kernel.cu
    force_kernel_launcher.cu
    force_bindings.cpp
)

target_include_directories(force_gpu PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(force_gpu PRIVATE pybind11::module)

set_target_properties(force_gpu PROPERTIES
    PREFIX ""  # important to not prefix with "lib"
    OUTPUT_NAME "force_gpu"
)
