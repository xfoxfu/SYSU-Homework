cmake_minimum_required(VERSION 3.10.0)
project(lab6 VERSION 0.1.0)

add_subdirectory(fmt)

find_package(CUDAToolkit REQUIRED 10.2)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)

add_library(common
    matrix.cpp
)

target_include_directories(common PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(common PUBLIC CUDA::cudart)
target_link_libraries(common PUBLIC fmt::fmt)

add_executable(task1 task1.cu)
target_link_libraries(task1 common)

add_executable(task2 task2.cu)
target_link_libraries(task2 common)

add_executable(task3 task3.cu)
target_link_libraries(task3 common cudnn)
