##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

# TODO(ldionne): This CMake testing infrastructure should be replaced with a
#                llvm-lit test suite.

add_custom_target(pstl-build-tests
    COMMENT "Build all the pstl tests.")

add_custom_target(check-pstl
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    USES_TERMINAL
    DEPENDS pstl-build-tests
    COMMENT "Build and run all the unit tests.")

add_library(test_stdlib INTERFACE)
target_include_directories(test_stdlib INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/../stdlib")
add_library(test_support INTERFACE)
target_include_directories(test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/pstl_testsuite")

file(GLOB_RECURSE UNIT_TESTS "*.pass.cpp")
foreach(_file IN LISTS UNIT_TESTS)
    get_filename_component(_test_name ${_file} NAME_WE)

    add_executable(${_test_name} "${_file}")
    target_compile_definitions(${_test_name} PRIVATE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)
    target_include_directories(${_test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${_test_name} PRIVATE oneDPL test_stdlib test_support)
    set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)
    if (PARALLELSTL_DEVICE_TYPE)
        add_test(NAME ${_test_name} COMMAND ${CMAKE_COMMAND} -E env SYCL_DEVICE_TYPE=${PARALLELSTL_DEVICE_TYPE} ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    else()
        add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    endif()
    add_dependencies(pstl-build-tests ${_test_name})
endforeach()
