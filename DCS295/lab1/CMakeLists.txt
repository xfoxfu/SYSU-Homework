cmake_minimum_required(VERSION 3.0.0)
project(lab1 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(lab1 
    src/main.cpp
    src/matrix_extended.cpp
    src/matrix_truncated.cpp
    src/matrix.cpp
    src/prod_standard.cpp
    src/prod_strassen.cpp
)

target_compile_features(lab1 PUBLIC cxx_std_17)
set_target_properties(lab1 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab1 PRIVATE -Werror -Wall -Wextra)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
