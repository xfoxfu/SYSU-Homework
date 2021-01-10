#
# This module is designed to find/handle mysql(client) library
#
# Requirements:
# - CMake >= 2.8.3 (for new version of find_package_handle_standard_args)
#
# The following variables will be defined for your use:
#   - MySQL_INCLUDE_DIRS  : mysql(client) include directory
#   - MySQL_LIBRARIES     : mysql(client) libraries
#   - MySQL_VERSION       : complete version of mysql(client) (x.y.z)
#   - MySQL_VERSION_MAJOR : major version of mysql(client)
#   - MySQL_VERSION_MINOR : minor version of mysql(client)
#   - MySQL_VERSION_PATCH : patch version of mysql(client)
#
# How to use:
#   1) Copy this file in the root of your project source directory
#   2) Then, tell CMake to search this non-standard module in your project directory by adding to your CMakeLists.txt:
#        set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
#   3) Finally call find_package(MySQL) once
#
# Here is a complete sample to build an executable:
#
#   set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
#
#   find_package(MySQL REQUIRED) # Note: name is case sensitive
#
#   add_executable(myapp myapp.c)
#   include_directories(${MySQL_INCLUDE_DIRS})
#   target_link_libraries(myapp ${MySQL_LIBRARIES})
#   # with CMake >= 3.0.0, the last two lines can be replaced by the following
#   target_link_libraries(myapp MySQL::MySQL) # Note: case also matters here
#


#=============================================================================
# Copyright (c) 2013-2020, julp
#
# Distributed under the OSI-approved BSD License
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#=============================================================================

# TODO:
# - mariadb support?
# - on Windows, lookup for related registry keys

cmake_minimum_required(VERSION 3.0.0)

# "As of MySQL 5.7.9, MySQL distributions contain a mysqlclient.pc file that provides information about MySQL configuration for use by the pkg-config command."
find_package(PkgConfig QUIET)

########## Private ##########
if(NOT DEFINED MySQL_PUBLIC_VAR_NS)
    set(MySQL_PUBLIC_VAR_NS "MySQL")
endif(NOT DEFINED MySQL_PUBLIC_VAR_NS)
if(NOT DEFINED MySQL_PRIVATE_VAR_NS)
    set(MySQL_PRIVATE_VAR_NS "_${MySQL_PUBLIC_VAR_NS}")
endif(NOT DEFINED MySQL_PRIVATE_VAR_NS)
if(NOT DEFINED PC_MySQL_PRIVATE_VAR_NS)
    set(PC_MySQL_PRIVATE_VAR_NS "_PC${MySQL_PRIVATE_VAR_NS}")
endif(NOT DEFINED PC_MySQL_PRIVATE_VAR_NS)

# Alias all MySQL_FIND_X variables to MySQL_FIND_X
# Workaround for find_package: no way to force case of variable's names it creates (I don't want to change MY coding standard)
set(${MySQL_PRIVATE_VAR_NS}_FIND_PKG_PREFIX "MySQL")
get_directory_property(${MySQL_PRIVATE_VAR_NS}_CURRENT_VARIABLES VARIABLES)
foreach(${MySQL_PRIVATE_VAR_NS}_VARNAME ${${MySQL_PRIVATE_VAR_NS}_CURRENT_VARIABLES})
    if(${MySQL_PRIVATE_VAR_NS}_VARNAME MATCHES "^${${MySQL_PRIVATE_VAR_NS}_FIND_PKG_PREFIX}")
        string(REGEX REPLACE "^${${MySQL_PRIVATE_VAR_NS}_FIND_PKG_PREFIX}" "${MySQL_PUBLIC_VAR_NS}" ${MySQL_PRIVATE_VAR_NS}_NORMALIZED_VARNAME ${${MySQL_PRIVATE_VAR_NS}_VARNAME})
        set(${${MySQL_PRIVATE_VAR_NS}_NORMALIZED_VARNAME} ${${${MySQL_PRIVATE_VAR_NS}_VARNAME}})
    endif(${MySQL_PRIVATE_VAR_NS}_VARNAME MATCHES "^${${MySQL_PRIVATE_VAR_NS}_FIND_PKG_PREFIX}")
endforeach(${MySQL_PRIVATE_VAR_NS}_VARNAME)

macro(_mysql_set_dotted_version VERSION_STRING)
    set(${MySQL_PUBLIC_VAR_NS}_VERSION "${VERSION_STRING}")
endmacro(_mysql_set_dotted_version)

########## Public ##########
if(PKG_CONFIG_FOUND)
    pkg_check_modules(${PC_MySQL_PRIVATE_VAR_NS} "mysqlclient" QUIET)
    if(${PC_MySQL_PRIVATE_VAR_NS}_FOUND)
		if(${PC_MySQL_PRIVATE_VAR_NS}_VERSION)
            _mysql_set_dotted_version("${${PC_MySQL_PRIVATE_VAR_NS}_VERSION}")
        endif(${PC_MySQL_PRIVATE_VAR_NS}_VERSION)
    endif(${PC_MySQL_PRIVATE_VAR_NS}_FOUND)
endif(PKG_CONFIG_FOUND)

find_program(${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE mysql_config)
if(${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE)
    execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --cflags                 OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_C_FLAGS)
    execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --version                OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_VERSION)
    execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --variable=pkglibdir     OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_LIBRARY_DIR)
    execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --variable=pkgincludedir OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_INCLUDE_DIR)
#     execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --plugindir              OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_PLUGIN_DIR)
#     execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --socket                 OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_SOCKET)
#     execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --port                   OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_PORT)
#     execute_process(OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND ${${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE} --libmysqld-libs         OUTPUT_VARIABLE ${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_LIBRARY_EMBEDDED)

    _mysql_set_dotted_version("${${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_VERSION}")
endif(${MySQL_PUBLIC_VAR_NS}_CONFIG_EXECUTABLE)

set(${MySQL_PRIVATE_VAR_NS}_COMMON_FIND_OPTIONS PATH_SUFFIXES mysql)

find_path(
    ${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR
    NAMES mysql_version.h
    ${${MySQL_PRIVATE_VAR_NS}_COMMON_FIND_OPTIONS}
    PATHS ${${PC_MySQL_PRIVATE_VAR_NS}_INCLUDE_DIRS} ${${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_INCLUDE_DIR}
)

if(WIN32)
    include(SelectLibraryConfigurations)
    # "On Windows, the static library is mysqlclient.lib and the dynamic library is libmysql.dll. Windows distributions also include libmysql.lib, a static import library needed for using the dynamic library."
    set(${MySQL_PRIVATE_VAR_NS}_POSSIBLE_NAMES "mysql" "mysqlclient")

    find_library(
        ${MySQL_PUBLIC_VAR_NS}_LIBRARY_RELEASE
        NAMES ${${MySQL_PRIVATE_VAR_NS}_POSSIBLE_NAMES}
        DOC "Release library for mysqlclient"
        ${${MySQL_PRIVATE_VAR_NS}_COMMON_FIND_OPTIONS}
    )
    # "Windows distributions also include a set of debug libraries. These have the same names as the nondebug libraries, but are located in the lib/debug library. You must use the debug libraries when compiling clients built using the debug C runtime."
    find_library(
        ${MySQL_PUBLIC_VAR_NS}_LIBRARY_DEBUG
        NAMES ${${MySQL_PRIVATE_VAR_NS}_POSSIBLE_NAMES}
        DOC "Debug library for mysqlclient"
        PATH_SUFFIXES mysql/debug debug
    )

    select_library_configurations("${MySQL_PUBLIC_VAR_NS}")
else(WIN32)
    # "On Unix (and Unix-like) sytems, the static library is libmysqlclient.a. The dynamic library is libmysqlclient.so on most Unix systems and libmysqlclient.dylib on OS X."
    find_library(
        ${MySQL_PUBLIC_VAR_NS}_LIBRARY
        NAMES mysqlclient
        PATHS ${${PC_MySQL_PRIVATE_VAR_NS}_LIBRARY_DIRS} ${${MySQL_PUBLIC_VAR_NS}_MySQL_CONFIG_LIBRARY_DIR}
        ${${MySQL_PRIVATE_VAR_NS}_COMMON_FIND_OPTIONS}
    )
endif(WIN32)

# Check find_package arguments
include(FindPackageHandleStandardArgs)
if(${MySQL_PUBLIC_VAR_NS}_FIND_REQUIRED AND NOT ${MySQL_PUBLIC_VAR_NS}_FIND_QUIETLY)
    find_package_handle_standard_args(
        ${MySQL_PUBLIC_VAR_NS}
        REQUIRED_VARS ${MySQL_PUBLIC_VAR_NS}_LIBRARY ${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR
        VERSION_VAR ${MySQL_PUBLIC_VAR_NS}_VERSION
    )
else(${MySQL_PUBLIC_VAR_NS}_FIND_REQUIRED AND NOT ${MySQL_PUBLIC_VAR_NS}_FIND_QUIETLY)
    find_package_handle_standard_args(${MySQL_PUBLIC_VAR_NS} "Could NOT find mysql(client)" ${MySQL_PUBLIC_VAR_NS}_LIBRARY ${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR)
endif(${MySQL_PUBLIC_VAR_NS}_FIND_REQUIRED AND NOT ${MySQL_PUBLIC_VAR_NS}_FIND_QUIETLY)

if(${MySQL_PUBLIC_VAR_NS}_FOUND)
    set(${MySQL_PUBLIC_VAR_NS}_LIBRARIES ${${MySQL_PUBLIC_VAR_NS}_LIBRARY})
    set(${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIRS ${${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR})
    if(CMAKE_VERSION VERSION_GREATER "3.0.0")
        if(NOT TARGET MySQL::MySQL)
            add_library(MySQL::MySQL UNKNOWN IMPORTED)
        endif(NOT TARGET MySQL::MySQL)
        if(${MySQL_PUBLIC_VAR_NS}_LIBRARY_RELEASE)
            set_property(TARGET MySQL::MySQL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(MySQL::MySQL PROPERTIES IMPORTED_LOCATION_RELEASE "${${MySQL_PUBLIC_VAR_NS}_LIBRARY_RELEASE}")
        endif(${MySQL_PUBLIC_VAR_NS}_LIBRARY_RELEASE)
        if(${MySQL_PUBLIC_VAR_NS}_LIBRARY_DEBUG)
            set_property(TARGET MySQL::MySQL APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(MySQL::MySQL PROPERTIES IMPORTED_LOCATION_DEBUG "${${MySQL_PUBLIC_VAR_NS}_LIBRARY_DEBUG}")
        endif(${MySQL_PUBLIC_VAR_NS}_LIBRARY_DEBUG)
        if(${MySQL_PUBLIC_VAR_NS}_LIBRARY)
            set_target_properties(MySQL::MySQL PROPERTIES IMPORTED_LOCATION "${${MySQL_PUBLIC_VAR_NS}_LIBRARY}")
        endif(${MySQL_PUBLIC_VAR_NS}_LIBRARY)
        set_target_properties(MySQL::MySQL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR}")
    endif(CMAKE_VERSION VERSION_GREATER "3.0.0")
endif(${MySQL_PUBLIC_VAR_NS}_FOUND)

mark_as_advanced(
    ${MySQL_PUBLIC_VAR_NS}_INCLUDE_DIR
    ${MySQL_PUBLIC_VAR_NS}_LIBRARY
)

########## <debug> ##########

if(${MySQL_PUBLIC_VAR_NS}_DEBUG)

    function(mysql_debug _VARNAME)
        if(DEFINED ${MySQL_PUBLIC_VAR_NS}_${_VARNAME})
            message("${MySQL_PUBLIC_VAR_NS}_${_VARNAME} = ${${MySQL_PUBLIC_VAR_NS}_${_VARNAME}}")
        else(DEFINED ${MySQL_PUBLIC_VAR_NS}_${_VARNAME})
            message("${MySQL_PUBLIC_VAR_NS}_${_VARNAME} = <UNDEFINED>")
        endif(DEFINED ${MySQL_PUBLIC_VAR_NS}_${_VARNAME})
    endfunction(mysql_debug)

    # IN (args)
    mysql_debug("FIND_REQUIRED")
    mysql_debug("FIND_QUIETLY")
    mysql_debug("FIND_VERSION")
    # OUT
    # Linking
    mysql_debug("INCLUDE_DIRS")
    mysql_debug("LIBRARIES")
    # Version
    mysql_debug("VERSION_MAJOR")
    mysql_debug("VERSION_MINOR")
    mysql_debug("VERSION_PATCH")
    mysql_debug("VERSION")

endif(${MySQL_PUBLIC_VAR_NS}_DEBUG)

########## </debug> ##########
