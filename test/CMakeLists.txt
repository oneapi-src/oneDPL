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
# rng_tests
set (ranlux_24_48_test.pass_timeout_debug "900") # 15min
set (ranlux_24_48_test.pass_timeout_release "720") # 12min

add_custom_target(build-onedpl-tests
    COMMENT "Build all oneDPL tests")

add_custom_target(run-onedpl-tests
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
    USES_TERMINAL
    DEPENDS build-onedpl-tests
    COMMENT "Build and run all oneDPL tests")

macro(onedpl_add_test test_source_file)
    file(RELATIVE_PATH _test_name "${CMAKE_CURRENT_SOURCE_DIR}" "${test_source_file}")
    string(REPLACE "\.cpp" "" _test_name ${_test_name})
    string(REPLACE "/" "-" _test_name ${_test_name})
    string(REPLACE "\.cpp" "" _test_name ${_test_name})

    add_executable(${_test_name} EXCLUDE_FROM_ALL "${test_source_file}")
    target_compile_definitions(${_test_name} PRIVATE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)
    if (MSVC)
        target_compile_options(${_test_name} PRIVATE /bigobj)
    endif()

    # oneDPL test harness may initialize a C++ iterator using iterator with different type
    # that may break code when using Intel(R) C++ Compiler Classic with -O3 flag on Linux
    if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL Intel)
        target_compile_options(${_test_name} PRIVATE $<$<CONFIG:Release>:-fno-strict-aliasing>)
    endif()

    target_include_directories(${_test_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/support")
    target_link_libraries(${_test_name} PRIVATE oneDPL)
    set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)

    add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})

    string(TOLOWER "${CMAKE_BUILD_TYPE}" _build_type_in_lower)
    if (DEFINED ${_test_name}_timeout_${_build_type_in_lower})
        message(STATUS
        "Timeout for ${_test_name} is set to ${${_test_name}_timeout_${_build_type_in_lower}}, "
        "it overrides user-defined timeout for CTest")
        set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${${_test_name}_timeout_${_build_type_in_lower}})
    elseif (DEFINED ${_test_name}_timeout_release)
        # Apply default (release) timeout
        message(STATUS
        "Timeout for ${_test_name} is set to ${${_test_name}_timeout_release}, "
        "it overrides user-defined timeout for CTest")
        set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${${_test_name}_timeout_release})
    endif()

    set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
    if (ONEDPL_DEVICE_TYPE)
        if (ONEDPL_DEVICE_TYPE MATCHES "FPGA")
            set(SYCL_DEVICE_FILTER *:acc)
        else()
            string(TOLOWER ${ONEDPL_DEVICE_TYPE} ONEDPL_DEVICE_TYPE)
            set(SYCL_DEVICE_FILTER ${ONEDPL_DEVICE_BACKEND}:${ONEDPL_DEVICE_TYPE})
        endif()
        set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER})
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
        if (NOT TARGET build-onedpl-${_label}-tests)
            add_custom_target(build-onedpl-${_label}-tests COMMENT "Build tests with label ${_label}")

            add_custom_target(run-onedpl-${_label}-tests
                COMMAND "${CMAKE_CTEST_COMMAND}" -L ^${_label}$$ --output-on-failure --no-label-summary
                USES_TERMINAL
                DEPENDS build-onedpl-${_label}-tests
                COMMENT "Build and run tests with label ${_label}")
        endif()
        add_dependencies(build-onedpl-${_label}-tests ${_test_name})
    endforeach()
    add_dependencies(build-onedpl-tests ${_test_name})
endmacro()

set(_skip_regex_for_not_dpcpp "(xpu_api/algorithms|xpu_api/functional|xpu_api/numerics)")
file(GLOB_RECURSE UNIT_TESTS "std/*.pass.cpp")
foreach (_file IN LISTS UNIT_TESTS)
    if (_file MATCHES "${_skip_regex_for_not_dpcpp}" AND NOT ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
        message(STATUS "Skip test ${_file} as it requires DPC++ backend")
        continue()
    endif()

    onedpl_add_test(${_file})
endforeach()
