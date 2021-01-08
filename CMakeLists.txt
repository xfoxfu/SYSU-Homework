cmake_minimum_required(VERSION 3.0.0)
project(bookman VERSION 0.1.0)

add_subdirectory(vendor/fmt)

find_package(PkgConfig)
pkg_check_modules(MYSQL REQUIRED mysqlclient>=8.0)

add_executable(bookman
    src/main.cpp
    src/ui.cpp
    src/mysqlclient.cpp
)

target_compile_features(bookman PUBLIC cxx_std_17)
set_target_properties(bookman PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(bookman PRIVATE -Wall -Wextra)

target_include_directories(bookman PRIVATE include)

target_include_directories(bookman PUBLIC ${MYSQL_INCLUDE_DIRS})
target_link_directories(bookman PUBLIC ${MYSQL_LIBRARY_DIRS})
target_link_libraries(bookman PUBLIC ${MYSQL_LIBRARIES})

target_link_libraries(bookman PUBLIC fmt::fmt)
