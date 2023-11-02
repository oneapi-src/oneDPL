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

option(ONEDPL_TEST_ENABLE_KT_ESIMD "Enable ESIMD-based kernel template tests")

function(_generate_test _target_name _test_path)
    add_executable(${_target_name} EXCLUDE_FROM_ALL ${_test_path})

    target_link_libraries(${_target_name} PRIVATE oneDPL)
    set_target_properties(${_target_name} PROPERTIES CXX_EXTENSIONS NO)

    add_test(NAME ${_target_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_target_name})
    set_tests_properties(${_target_name} PROPERTIES SKIP_RETURN_CODE 77)
    if (DEFINED DEVICE_SELECTION_LINE)
        set_tests_properties(${_target_name} PROPERTIES ENVIRONMENT ${DEVICE_SELECTION_LINE})
    endif()

    add_custom_target(run-${_target_name}
        COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_target_name}$$ --output-on-failure --no-label-summary
        USES_TERMINAL
        DEPENDS ${_target_name}
        COMMENT "Build and run test ${_target_name}")
endfunction(_generate_test)

if (ONEDPL_TEST_ENABLE_KT_ESIMD)
    add_custom_target(build-esimd-tests COMMENT "Build all ESIMD-based kernel template tests")
    add_custom_target(run-esimd-tests
        COMMAND "${CMAKE_CTEST_COMMAND}" -R "^run-esimd-tests$" --output-on-failure --no-label-summary
        DEPENDS build-esimd-tests
        COMMENT "Build and run all ESIMD tests")

    # Generate key-value sort test
    set(_target_name "esimd_radix_sort_by_key")
    set(_test_path "esimd_radix_sort_by_key.cpp")
    _generate_test(${_target_name} ${_test_path})
    add_dependencies(build-esimd-tests ${_target_name})
    add_dependencies(run-esimd-tests ${_target_name})

    # Generate key sort tests
    set(_key_type_all "char" "uint16_t" "int" "uint64_t" "float" "double")
    set(_data_per_work_item_all "32" "64" "96" "128" "160" "192" "224" "256" "288" "320" "352" "384" "416" "448" "480" "512")
    set(_work_group_size_all "32" "64")
    foreach (_key_type ${_key_type_all})
        foreach (_data_per_work_item ${_data_per_work_item_all})
            foreach (_work_group_size ${_work_group_size_all})
                string(REPLACE "_t" "" _key_type_short ${_key_type})
                set(_target_name "esimd_radix_sort_${_key_type_short}_dpwi${_data_per_work_item}_wgs${_work_group_size}")
                set(_test_path "esimd_radix_sort.cpp")
                _generate_test(${_target_name} ${_test_path})
                add_dependencies(build-esimd-tests ${_target_name})
                add_dependencies(run-esimd-tests ${_target_name})
                target_compile_definitions(${_target_name} PRIVATE TEST_DATA_TYPE=${_key_type})
                target_compile_definitions(${_target_name} PRIVATE TEST_DATA_PER_WORK_ITEM=${_data_per_work_item})
                target_compile_definitions(${_target_name} PRIVATE TEST_WORK_GROUP_SIZE=${_work_group_size})
            endforeach()
        endforeach()
    endforeach()
endif()
