# TODO implement me

include_directories("${PROJECT_SOURCE_DIR}/include"
                    ${CMAKE_CURRENT_LIST_DIR})

add_definitions(-DPSTL_STANDALONE_TESTS)

macro(add_pass_test test_source)
    string(REPLACE ".cpp" "" test_name ${test_source})
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} ${TBB_LIBRARIES})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_subdirectory(std)
