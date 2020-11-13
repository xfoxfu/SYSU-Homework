cmake_minimum_required(VERSION 3.0.0)
project(lab4 VERSION 0.1.0)

##### Task 1: Matrix Production #####

add_executable(matrix 
    src/main.cpp
    src/matrix_transposed.cpp
    src/matrix.cpp
    src/prod_standard.cpp
)

target_compile_features(matrix PUBLIC cxx_std_17)
set_target_properties(matrix PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(matrix PRIVATE -Wall -Wextra)
