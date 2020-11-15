cmake_minimum_required(VERSION 3.0.0)
project(graphics VERSION 0.1.0)

add_executable(graphics
    main.cpp
    PathTracer.cpp
    ray.cpp
    sphere.cpp
    hitable_list.cpp
    camera.cpp
    lambertian.cpp
    metal.cpp
    vec3.cpp
    dielectric.cpp
)

target_compile_features(graphics PUBLIC cxx_std_17)
set_target_properties(graphics PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(graphics PRIVATE -Wall -Wextra)
