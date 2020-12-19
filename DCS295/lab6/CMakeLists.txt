cmake_minimum_required(VERSION 3.10.0)
project(lab6 VERSION 0.1.0)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)

add_library(common
    matrix.cpp
    matrix_transposed.cpp
    prod_kernel.cu
    prod_cuda.cu
)

target_include_directories(common PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(common PUBLIC CUDA::cudart CUDA::cublas)

add_executable(cuda main_cuda.cpp)
target_link_libraries(cuda common)

add_executable(cuomp main_cuomp.cpp)
target_link_libraries(cuomp common)

add_executable(cublas main_cublas.cpp)
target_link_libraries(cublas common)
