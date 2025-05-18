cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(lab4 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(LAB_SOURCE_FILES
        src/helper_cuda.h
        src/obj_structures.cpp
        src/obj_structures.h
        src/point_alignment.cu
)

add_executable(point_alignment "${LAB_SOURCE_FILES}")

set(LAB_LINK_LIBRARIES
        CUDA::cublas
        CUDA::cusolver
)
target_link_libraries(point_alignment "${LAB_LINK_LIBRARIES}")

# --keep: save PTX files in build directory
# -src-in-ptx: Interleave source in PTX.
# --generate-line-info: Generate line-number information for device code.
#       - useful for profiling, and for showing location of errors in compute-sanitizer
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info>)
