cmake_minimum_required(VERSION 3.0.0)
project(lab5 VERSION 0.1.0)

########## parallel_for ##########

add_library(parallel_for
    lib/parallel_for.cpp
)

target_compile_features(parallel_for PUBLIC cxx_std_17)
set_target_properties(parallel_for PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(parallel_for PRIVATE -Wall -Wextra)

target_include_directories(parallel_for PRIVATE include)

########## lab5 OpenMP ##########

add_executable(lab5-omp omp.cpp)

target_compile_features(lab5-omp PUBLIC cxx_std_17)
set_target_properties(lab5-omp PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab5-omp PRIVATE -Wall -Wextra)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lab5-omp PUBLIC OpenMP::OpenMP_CXX)
endif()

########## lab5 parallel_for ##########

add_executable(lab5-pfor pfor.cpp parallel_for_closure.cpp)

target_compile_features(lab5-pfor PUBLIC cxx_std_17)
set_target_properties(lab5-pfor PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab5-pfor PRIVATE -Wall -Wextra)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lab5-pfor PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lab5-pfor PUBLIC parallel_for)
target_include_directories(lab5-pfor PRIVATE include)

########## lab5 MPI ##########

add_executable(lab5-mpi mpi.cpp parallel_for_closure.cpp)

target_compile_features(lab5-mpi PUBLIC cxx_std_17)
set_target_properties(lab5-mpi PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab5-mpi PRIVATE -Wall -Wextra)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lab5-mpi PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lab5-mpi PUBLIC parallel_for)
target_include_directories(lab5-mpi PRIVATE include)

find_package(MPI REQUIRED)
target_compile_options(lab5-mpi PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(lab5-mpi PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
target_include_directories(lab5-mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})
