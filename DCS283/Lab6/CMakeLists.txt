cmake_minimum_required(VERSION 3.0.0)
project(lab6 VERSION 0.1.0)

find_package(PkgConfig)
pkg_check_modules(MYSQL REQUIRED mysqlclient>=8.0)

set(CMAKE_CXX_STANDARD 17)

add_executable(lab6 main.cpp)
target_include_directories(lab6 PUBLIC ${MYSQL_INCLUDE_DIRS})
target_link_libraries(lab6 PUBLIC ${MYSQL_LIBRARY})
