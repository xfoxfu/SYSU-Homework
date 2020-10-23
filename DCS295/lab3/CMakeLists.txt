cmake_minimum_required(VERSION 3.0.0)
project(lab3 VERSION 0.1.0)

##### Task 1: Matrix Production #####

add_executable(matrix 
    src/matrix/main.cpp
    src/matrix/matrix_extended.cpp
    src/matrix/matrix_transposed.cpp
    src/matrix/matrix_truncated.cpp
    src/matrix/matrix.cpp
    src/matrix/prod_mpi.cpp
    src/matrix/prod_standard.cpp
    src/matrix/prod_strassen.cpp
)

target_compile_features(matrix PUBLIC cxx_std_17)
set_target_properties(matrix PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(matrix PRIVATE -Werror -Wall -Wextra)

##### Task 2: Array Sum (Trivial) #####

add_executable(sum_trivial src/sum/trivial.cpp)

target_compile_features(sum_trivial PUBLIC cxx_std_17)
set_target_properties(sum_trivial PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(sum_trivial PRIVATE -Werror -Wall -Wextra)

##### Task 2: Array Sum (Grouped) #####

add_executable(sum_grouped src/sum/grouped.cpp)

target_compile_features(sum_grouped PUBLIC cxx_std_17)
set_target_properties(sum_grouped PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(sum_grouped PRIVATE -Werror -Wall -Wextra)

##### Task 3: Solve Quadratic Equation #####

add_executable(equation src/equation/main.cpp)

target_compile_features(equation PUBLIC cxx_std_17)
set_target_properties(equation PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(equation PRIVATE -Werror -Wall -Wextra)

##### Task 4: Monte Carlo #####

add_executable(area src/area/main.cpp)

target_compile_features(area PUBLIC cxx_std_17)
set_target_properties(area PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(area PRIVATE -Werror -Wall -Wextra)
