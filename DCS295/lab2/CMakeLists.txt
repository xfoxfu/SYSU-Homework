cmake_minimum_required(VERSION 3.0.0)
project(lab2 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(lab2 
    src/main.cpp
    src/matrix_extended.cpp
    src/matrix_transposed.cpp
    src/matrix_truncated.cpp
    src/matrix.cpp
    src/prod_mpi.cpp
    src/prod_standard.cpp
    src/prod_strassen.cpp
)

find_package(MPI REQUIRED)

target_compile_features(lab2 PUBLIC cxx_std_17)
set_target_properties(lab2 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab2 PRIVATE -Werror -Wall -Wextra)
target_compile_options(lab2 PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(lab2 ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
target_include_directories(lab2 PRIVATE ${MPI_CXX_INCLUDE_PATH})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
