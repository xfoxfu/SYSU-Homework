cmake_minimum_required(VERSION 3.0.0)
project(lab4 VERSION 0.1.0)

##### Task 3: parallel_for library #####

add_library(parallel_for SHARED
    lib/parallel_for.cpp
)

target_compile_features(parallel_for PUBLIC cxx_std_17)
set_target_properties(parallel_for PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(parallel_for PRIVATE -Wall -Wextra)

target_include_directories(parallel_for PRIVATE include)

##### Task 1: Matrix Production #####

add_executable(matrix 
    src/main.cpp
    src/matrix_transposed.cpp
    src/matrix.cpp
    src/prod_standard.cpp
    src/prod_omp.cpp
    src/prod_pfor.cpp
)

target_compile_features(matrix PUBLIC cxx_std_17)
set_target_properties(matrix PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(matrix PRIVATE -Wall -Wextra)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(matrix PUBLIC parallel_for)
target_include_directories(matrix PRIVATE include)
