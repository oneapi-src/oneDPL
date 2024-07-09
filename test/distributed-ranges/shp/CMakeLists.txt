# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

if(${CMAKE_VERSION} VERSION_LESS "3.26")
    message(WARNING "disabled distributed ranges tests, use cmake>=3.26")
    return()
else ()
    message(STATUS "suitable for distributed ranges cmake version ${CMAKE_VERSION} found")
endif()

find_package(fmt QUIET)
if(fmt_FOUND)
    if(fmt_VERSION VERSION_GREATER_EQUAL 8.1 AND fmt_VERSION VERSION_LESS 10)
        message(STATUS "suitable for distributed ranges libfmt version ${fmt_VERSION} found")
    else()
        message(WARNING "libfmt version ${fmt_VERSION} is not >= 8.1 and < 10, skipping distributed ranges tests")
        return()

    endif()
else()
    message(WARNING "libfmt not found, install libfmt-dev 8.x or 9.x to enable distributed ranges tests")
    return()
endif()

find_package(GTest)
if (NOT GTest_FOUND)
  message(WARNING "disabled distributed ranges tests, install libgtest-dev or other package with GTest")
  return()
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-tests
  shp-tests.cpp 
  ../common/all.cpp 
  ../common/copy.cpp 
  ../common/counted.cpp
  ../common/distributed_vector.cpp 
  ../common/drop.cpp
  ../common/enumerate.cpp
  ../common/fill.cpp 
  ../common/for_each.cpp
  ../common/iota.cpp
  # ../common/iota_view.cpp # compile fail, even in distributed-ranges project
  ../common/reduce.cpp
  ../common/sort.cpp
  ../common/subrange.cpp
  ../common/take.cpp 
  ../common/transform.cpp 
  ../common/transform_view.cpp
  ../common/zip.cpp
  ../common/zip_local.cpp 
  algorithms.cpp
  containers.cpp 
  copy.cpp 
  detail.cpp 
  fill.cpp 
  transform.cpp
  )

add_executable(shp-tests-3 shp-tests.cpp containers-3.cpp copy-3.cpp)

foreach(test-exec IN ITEMS shp-tests shp-tests-3)
  target_link_libraries(${test-exec} PRIVATE GTest::gtest_main fmt::fmt oneDPL)
  target_link_options(${test-exec} PRIVATE -qopenmp)
endforeach()

add_custom_target(shp-all-tests)

function(add_shp_ctest test_name name)
  add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL SHP)
  add_dependencies(shp-all-tests ${name})
endfunction()

# Sort excluded due to https://github.com/oneapi-src/distributed-ranges/issues/785
add_shp_ctest(shp-tests shp-tests)
add_shp_ctest(shp-tests-3 shp-tests --devicesCount 3 --gtest_filter=-Sort*:ShpTests.Sort*)
add_shp_ctest(shp-tests-3-only shp-tests-3 --devicesCount 3)
