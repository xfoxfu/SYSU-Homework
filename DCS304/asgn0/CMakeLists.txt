cmake_minimum_required(VERSION 3.0.0)
project(main VERSION 0.1.0)

file(GLOB "src/*.cpp")

add_executable(main
    src/main.cpp
    src/glad.c
)
include_directories(include)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(main glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(main OpenGL::GL)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
