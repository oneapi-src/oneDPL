

set(TESTS
        "test_is_sorted"
        "test_copy_move"
        "test_transform_scan"
        "test_iterators"
        "test_find_end"
        "test_memory"
        "test_partition_copy"
        "test_all_of"
        "test_scan"
        "test_sort"
        "test_mismatch"
        "test_transform_unary"
        "test_replace_copy"
        "test_fill"
        "test_numeric"
        "test_minmax_element"
        "test_uninitialized_copy_move"
        "test_count"
        "test_equal"
        "test_for_each"
        "test_uninitialized_fill_destroy"
        "test_generate"
        "test_copy_if"
        "test_transform_reduce"
        "test_find"
        "test_uninitialized_construct"
        "test_find_if"
        "test_reduce"
        "test_remove_copy"
        "test_unique_copy_equal"
        "test_transform_binary"
        "test_algorithm"
        "test_none_of"
        "test_any_of"
        "test_adjacent_find")

add_library(TestsCommon INTERFACE)
target_include_directories(TestsCommon INTERFACE "${ParallelSTL_SOURCE_DIR}")

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} PUBLIC TestsCommon ParallelSTL)
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

