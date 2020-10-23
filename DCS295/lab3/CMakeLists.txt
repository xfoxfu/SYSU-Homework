cmake_minimum_required(VERSION 3.0.0)
project(lab3 VERSION 0.1.0)

##### Task 1: Matrix Production #####

add_executable(matrix 
    src/matrix/main.cpp
    src/matrix/matrix_transposed.cpp
    src/matrix/matrix.cpp
    src/matrix/prod_standard.cpp
)

target_compile_features(matrix PUBLIC cxx_std_17)
set_target_properties(matrix PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(matrix PRIVATE -Wall -Wextra)

##### Task 2: Array Sum #####

add_executable(sum src/sum/main.cpp)

option(SUM_USE_ATOMIC "use atomic for sum" OFF)
if(SUM_USE_ATOMIC)
    target_compile_definitions(sum PRIVATE USE_ATOMIC)
endif(SUM_USE_ATOMIC)

target_compile_features(sum PUBLIC cxx_std_17)
set_target_properties(sum PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(sum PRIVATE -Wall -Wextra)

##### Task 3: Solve Quadratic Equation #####

add_executable(equation src/equation/main.cpp src/equation/semaphore.cpp)

target_compile_features(equation PUBLIC cxx_std_17)
set_target_properties(equation PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(equation PRIVATE -Wall -Wextra)

##### Task 4: Monte Carlo #####

add_executable(area src/area/main.cpp)

target_compile_features(area PUBLIC cxx_std_17)
set_target_properties(area PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(area PRIVATE -Wall -Wextra)
