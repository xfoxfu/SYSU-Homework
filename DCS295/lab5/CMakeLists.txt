cmake_minimum_required(VERSION 3.0.0)
project(lab5 VERSION 0.1.0)

########## parallel_for ##########

add_library(parallel_for
    ../lab4/lib/parallel_for.cpp
)

target_compile_features(parallel_for PUBLIC cxx_std_17)
set_target_properties(parallel_for PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(parallel_for PRIVATE -Wall -Wextra)

target_include_directories(parallel_for PRIVATE include)

########## lab5 ##########

add_executable(lab5 main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lab5 PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lab5 PUBLIC parallel_for)
target_include_directories(lab5 PRIVATE include)

