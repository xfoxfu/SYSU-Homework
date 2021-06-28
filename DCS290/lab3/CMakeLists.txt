cmake_minimum_required(VERSION 3.0.0)
project(lab3 VERSION 0.1.0)

add_executable(lab3
    main.cpp
    ast.cpp
    error.cpp
    lexer.cpp
    parser.cpp
    span.cpp
    token.cpp
)

target_include_directories(lab3 PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_features(lab3 PUBLIC cxx_std_17)
set_target_properties(lab3 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab3 PRIVATE -Wall -Wextra)
