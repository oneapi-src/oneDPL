
function(generate_esimd_radix_sort_tests)
    add_custom_target(build-esimd-tests COMMENT "Build all ESIMD tests")
    add_custom_target(run-esimd-tests
        COMMAND "${CMAKE_CTEST_COMMAND}" -R ${_test_base_name} --output-on-failure --no-label-summary
        USES_TERMINAL
        DEPENDS build-esimd-tests
        COMMENT "Build and run all ESIMD tests")

    # Add test for ...esimd::radix_sort_by_key(...)
    set(_test_name "esimd_radix_sort_by_key")
    add_executable(${_test_name} EXCLUDE_FROM_ALL "${_test_name}.cpp")
    target_link_libraries(${_test_name} PRIVATE oneDPL)
    set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)
    add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
    set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
    if (DEFINED DEVICE_SELECTION_LINE)
        set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT ${DEVICE_SELECTION_LINE})
    endif()
    add_custom_target(run-${_test_name}
        COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_test_name}$$ --output-on-failure --no-label-summary
        USES_TERMINAL
        DEPENDS ${_test_name}
        COMMENT "Build and run test ${_test_name}")
    add_dependencies(build-esimd-tests ${_test_name})
    add_dependencies(run-esimd-tests ${_test_name})

    # Generate tests for ...esimd::radix_sort(...)
    set(_test_base_name "esimd_radix_sort")
    set(_test_source_file "esimd_radix_sort.cpp")
    # set(_key_type_all "char" "int8_t" "uint8_t" "int16_t" "uint16_t" "int" "uint32_t" "int64_t" "uint64_t" "float" "double")
    # set(_work_group_size_all "16" "24" "32" "48" "56" "64")
    set(_key_type_all "char" "uint16_t" "int" "uint64_t" "float" "double")
    set(_data_per_work_item_all "32" "64" "96" "128" "160" "192" "224" "256" "288" "320" "352" "384" "416" "448" "480" "512")
    set(_work_group_size_all "32" "64")

    foreach (_key_type ${_key_type_all})
        foreach (_data_per_work_item ${_data_per_work_item_all})
            foreach (_work_group_size ${_work_group_size_all})
                string(REPLACE "_t" "" _key_type_short ${_key_type})
                set(_test_name ${_test_base_name}_${_key_type_short}_dpwi${_data_per_work_item}_wgs${_work_group_size})
                add_executable(${_test_name} EXCLUDE_FROM_ALL "${_test_source_file}")

                target_link_libraries(${_test_name} PRIVATE oneDPL)
                set_target_properties(${_test_name} PROPERTIES CXX_EXTENSIONS NO)

                add_test(NAME ${_test_name} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${_test_name})
                set_tests_properties(${_test_name} PROPERTIES SKIP_RETURN_CODE 77)
                if (DEFINED DEVICE_SELECTION_LINE)
                    set_tests_properties(${_test_name} PROPERTIES ENVIRONMENT ${DEVICE_SELECTION_LINE})
                endif()

                target_compile_definitions(${_test_name} PRIVATE TEST_DATA_TYPE=${_key_type})
                target_compile_definitions(${_test_name} PRIVATE TEST_DATA_PER_WORK_ITEM=${_data_per_work_item})
                target_compile_definitions(${_test_name} PRIVATE TEST_WORK_GROUP_SIZE=${_work_group_size})

                add_custom_target(run-${_test_name}
                    COMMAND "${CMAKE_CTEST_COMMAND}" -R ^${_test_name}$$ --output-on-failure --no-label-summary
                    USES_TERMINAL
                    DEPENDS ${_test_name}
                    COMMENT "Build and run test ${_test_name}")
                add_dependencies(build-esimd-tests ${_test_name})
                add_dependencies(run-esimd-tests ${_test_name})
            endforeach()
        endforeach()
    endforeach()
endfunction()

generate_esimd_radix_sort_tests()
