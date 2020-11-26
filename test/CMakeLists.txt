##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) 2017-2020 Intel Corporation
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

add_custom_target(build-all
    COMMENT "Build all the pstl tests.")

add_custom_target(run-all
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    USES_TERMINAL
    DEPENDS build-all
    COMMENT "Build and run all the unit tests.")

add_library(test_support INTERFACE)
target_include_directories(test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/pstl_testsuite")

macro(onedpl_add_test test_source_file)
    get_filename_component(_test_name ${test_source_file} NAME)
    string(REPLACE "\.cpp" "" _test_name ${_test_name})

    add_executable(${_test_name} "${test_source_file}")
    target_compile_definitions(${_test_name} PRIVATE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)
    if (MSVC)
        target_compile_options(${_test_name} PRIVATE /bigobj)
    endif()
    target_include_directories(${_test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${_test_name} PRIVATE oneDPL test_support)
    set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)
    if (ONEDPL_DEVICE_TYPE)
        set(SYCL_DEVICE_TYPE ${ONEDPL_DEVICE_TYPE})
        if (ONEDPL_DEVICE_TYPE MATCHES "FPGA")
            set(SYCL_DEVICE_TYPE ACC)
        endif()
        add_test(NAME ${_test_name} COMMAND ${CMAKE_COMMAND} -E env SYCL_DEVICE_TYPE=${SYCL_DEVICE_TYPE} ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    else()
        add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    endif()

    add_custom_target(run-${_test_name}
        COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_test_name}$$ --output-on-failure --no-label-summary
        USES_TERMINAL
        DEPENDS ${_test_name}
        COMMENT "Build and run test ${_test_name}")

    # Add labels and group targets
    file(RELATIVE_PATH _test_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${test_source_file})
    get_filename_component(_test_rel_path ${_test_rel_path} DIRECTORY)
    if (_test_rel_path)
        string(REPLACE "/" ";" _test_labels ${_test_rel_path})
        set_tests_properties(${_test_name} PROPERTIES LABELS "${_test_labels}")
    endif()

    foreach (_label ${_test_labels})
        if (NOT TARGET build-${_label})
            add_custom_target(build-${_label} COMMENT "Build tests with label ${_label}")

            add_custom_target(run-${_label}
                COMMAND "${CMAKE_CTEST_COMMAND}" -L ^${_label}$$ --output-on-failure --no-label-summary
                USES_TERMINAL
                DEPENDS build-${_label}
                COMMENT "Build and run tests with label ${_label}")
        endif()
        add_dependencies(build-${_label} ${_test_name})
    endforeach()
    add_dependencies(build-all ${_test_name})
endmacro()

#add_subdirectory(pstl_testsuite)
#add_subdirectory(extenstions_testsuite)
#if (ONEDPL_BACKEND MATCHES "^(sycl|sycl_only)$")

set(_skip_regex_for_not_sycl "(rng_testsuite|extensions_testsuite)")
file(GLOB_RECURSE UNIT_TESTS "*.pass.cpp")
foreach (_file IN LISTS UNIT_TESTS)
    if (_file MATCHES "${_skip_regex_for_not_sycl}" AND NOT ONEDPL_BACKEND MATCHES "^(sycl|sycl_only)$")
        message(STATUS "Skip test ${_file} as it requires SYCL backend")
        continue()
    endif()

    onedpl_add_test(${_file})
endforeach()
