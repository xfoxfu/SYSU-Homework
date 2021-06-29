cmake_minimum_required(VERSION 3.0.0)
project(lab4 VERSION 0.1.0)

add_executable(lab4
    ast.cpp
    error.cpp
    lexer.cpp
    main.cpp
    parser.cpp
    semantic.cpp
    span.cpp
    token.cpp
)

target_include_directories(lab4 PRIVATE ${CMAKE_SOURCE_DIR})

target_compile_features(lab4 PUBLIC cxx_std_17)
set_target_properties(lab4 PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lab4 PRIVATE -Wall -Wextra)
target_compile_definitions(lab4 PRIVATE FMT_HEADER_ONLY)
