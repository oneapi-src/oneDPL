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

if(${CMAKE_VERSION} VERSION_LESS "3.26")
    message(WARNING "disabled distributed ranges tests, use cmake>=3.26")
    return()
else ()
    message(STATUS "suitable for distributed ranges cmake version ${CMAKE_VERSION} found")
endif()

add_library(dr_shp INTERFACE)
add_library(DR::shp ALIAS dr_shp)

find_package(fmt QUIET)
if(fmt_FOUND)
    if(fmt_VERSION VERSION_GREATER_EQUAL 8.1 AND fmt_VERSION VERSION_LESS 10)
      message(STATUS "suitable for distributed ranges libfmt version ${fmt_VERSION} found")
      target_compile_options(dr_shp INTERFACE -DUSE_FMT)
    else()
        message(WARNING "libfmt version ${fmt_VERSION} unsuitable, no detailed messages in dr tests")
    endif()
else()
    message(WARNING "libfmt not found, no detailed messages in dr tests")
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
  ../common/exclusive_scan.cpp
  ../common/fill.cpp 
  ../common/for_each.cpp
  ../common/inclusive_scan.cpp
  ../common/iota.cpp
  ../common/iota_view.cpp
  ../common/reduce.cpp
  ../common/sort.cpp
  ../common/subrange.cpp
  ../common/sycl_utils.cpp
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
  target_link_libraries(${test-exec} PRIVATE GTest::gtest_main oneDPL)
  if(fmt_FOUND)
    target_link_libraries(${test-exec} fmt::fmt)
  endif()
  target_link_options(${test-exec} PRIVATE -qopenmp)
endforeach()

add_custom_target(shp-all-tests)

function(add_shp_ctest test_name name)
  add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL SHP)
  add_dependencies(shp-all-tests ${name})
endfunction()

# sort: https://github.com/oneapi-src/distributed-ranges/issues/785
# ExclusiveScan.empty: https://github.com/oneapi-src/distributed-ranges/issues/791
# IotaView.ZipWithDR: https://github.com/oneapi-src/distributed-ranges/issues/792
set(test-exclusions Sort*:ShpTests.Sort*:ExclusiveScan*empty:IotaView*ZipWithDR)

add_shp_ctest(shp-tests shp-tests --gtest_filter=-${test-exclusions})
add_shp_ctest(shp-tests-3 shp-tests --devicesCount 3 --gtest_filter=-${test-exclusions})
add_shp_ctest(shp-tests-3-only shp-tests-3 --devicesCount 3)
