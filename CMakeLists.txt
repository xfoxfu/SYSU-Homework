cmake_minimum_required(VERSION 3.0.0)
project(bookman VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_subdirectory(vendor/fmt)
find_package(MySQL REQUIRED)

add_executable(main
    src/main.cpp
    src/mysqlclient.cpp
    src/refund.cpp
    src/sell.cpp
    src/stock.cpp
    src/table_printer.cpp
    src/test.cpp
    src/ui.cpp
)

target_compile_features(main PUBLIC cxx_std_17)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(main PRIVATE -Wall -Wextra)

target_include_directories(main PUBLIC
    include
    ${MYSQL_INCLUDE_DIRS}
)
target_link_directories(main PUBLIC
    ${MYSQL_LIBRARY_DIRS}
)
target_link_libraries(main PUBLIC
    fmt::fmt
    ${MYSQL_LIBRARIES}
)
