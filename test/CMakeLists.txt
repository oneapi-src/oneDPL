##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
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

macro(onedpl_add_test test_source_file)
    get_filename_component(_test_name ${test_source_file} NAME)
    string(REPLACE "\.cpp" "" _test_name ${_test_name})

    add_executable(${_test_name} "${test_source_file}")
    target_compile_definitions(${_test_name} PRIVATE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)
    if (MSVC)
        target_compile_options(${_test_name} PRIVATE /bigobj)
    endif()

    # oneDPL test harness may initialize a C++ iterator using iterator with different type
    # that may break code when using Intel(R) C++ Compiler Classic with -O3 flag on Linux
    if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL Intel)
        target_compile_options(${_test_name} PRIVATE $<$<CONFIG:Release>:-fno-strict-aliasing>)
    endif()

    target_include_directories(${_test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(${_test_name} PRIVATE oneDPL)
    set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)

    add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
    if (ONEDPL_DEVICE_TYPE)
        # TODO: Consider replacement of SYCL_DEVICE_TYPE with a new variable.
        #       See for details: https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md
        set(SYCL_DEVICE_TYPE ${ONEDPL_DEVICE_TYPE})
        if (ONEDPL_DEVICE_TYPE MATCHES "FPGA")
            set(SYCL_DEVICE_TYPE ACC)
        endif()
        set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT SYCL_DEVICE_TYPE=${SYCL_DEVICE_TYPE})
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

set(_skip_regex_for_not_dpcpp "(xpu_api)")
file(GLOB_RECURSE UNIT_TESTS "*.pass.cpp")
foreach (_file IN LISTS UNIT_TESTS)
    if (_file MATCHES "${_skip_regex_for_not_dpcpp}" AND NOT ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
        message(STATUS "Skip test ${_file} as it requires DPC++ backend")
        continue()
    endif()

    onedpl_add_test(${_file})
endforeach()
