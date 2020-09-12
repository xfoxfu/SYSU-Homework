cmake_minimum_required(VERSION 3.10.0)
project(main VERSION 0.1.0)

# STB_IMAGE
add_library(STB_IMAGE "vendor/stb_image/stb_image.cpp")

# GLAD
add_library(GLAD "vendor/glad/glad.c")

# main
add_executable(main
    src/main.cpp
    src/shader.cpp
    src/vao.cpp
    src/vbo.cpp
)
include_directories(include)

target_compile_features(main PUBLIC cxx_std_17)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(main PRIVATE -Werror -Wall -Wextra)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(main glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(main OpenGL::GL)
find_package(glm REQUIRED)
target_link_libraries(main glm::glm)
target_link_libraries(main GLAD)
target_link_libraries(main STB_IMAGE)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


function(add_assets)
    foreach(src ${ARGN})
        configure_file(assets/${src} assets/${src} COPYONLY)
    endforeach(src)
endfunction(add_assets)

add_assets(
    fragment.frag 
    vertex.vert
    container.jpg
)
