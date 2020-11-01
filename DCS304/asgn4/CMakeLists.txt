cmake_minimum_required(VERSION 3.10.0)
project(main VERSION 0.1.0)

# main
add_executable(main
    geometry.cpp
    main.cpp
    model.cpp
    our_gl.cpp
    tgaimage.cpp
)
include_directories(include)

target_compile_features(main PUBLIC cxx_std_17)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(main PRIVATE -Wall -Wextra)

# find_package(glfw3 3.3 REQUIRED)
# target_link_libraries(main glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(main OpenGL::GL)
# find_package(glm REQUIRED)
# target_link_libraries(main glm::glm)
find_package(OpenCV REQUIRED)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


function(add_assets)
    foreach(src ${ARGN})
        configure_file(obj/${src} obj/${src} COPYONLY)
    endforeach(src)
endfunction(add_assets)

add_assets(
    diablo3_pose.obj
    diablo3_pose_diffuse.tga
    diablo3_pose_glow.tga
    diablo3_pose_nm.tga
    diablo3_pose_nm_tangent.tga
    diablo3_pose_spec.tga
    floor.obj
    floor_diffuse.tga
    floor_nm_tangent.tga
    floor_spec.tga
    grid.tga
    readme.txt
)
