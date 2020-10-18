cmake_minimum_required(VERSION 3.0.0)
project(lab6 VERSION 0.1.0)

find_package(PkgConfig)
pkg_check_modules(MySQL REQUIRED mysqlclient>=5.7)

add_executable(lab6 main.cpp)
target_include_directories(lab6 PUBLIC ${MYSQL_INCLUDE_DIRS})
target_link_libraries(lab6 PUBLIC ${MYSQL_LIBRARY})
