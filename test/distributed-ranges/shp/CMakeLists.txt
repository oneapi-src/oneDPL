# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

include(CheckIncludeFileCXX)
# check_include_file_cxx("format" CXX_FORMAT_SUPPORT)
# if (NOT CXX_FORMAT_SUPPORT)
  find_package(fmt QUIET)
  if (NOT fmt_FOUND)
    message(WARNING "disabled distributed ranges tests, install fmt library or g++>=13 or clang>=17")
    return()
  endif ()
# endif()

check_include_file_cxx("cxxopts.hpp" CXX_OPTS_SUPPORT)
if (NOT CXX_OPTS_SUPPORT)
  message(WARNING "disabled distributed ranges tests, install libcxxopts-dev or other package with cxxopts")
  return()
endif ()

find_package(GTest)
if (NOT GTest_FOUND)
  message(WARNING "disabled distributed ranges tests, install libgtest-dev or other package with GTest")
  return()
endif ()

find_package(MKL REQUIRED)

# include(GoogleTest)

add_library(dr_shp INTERFACE)
add_library(DR::shp ALIAS dr_shp)

target_include_directories(dr_shp INTERFACE . vendor ../../../include)
target_compile_definitions(dr_shp INTERFACE USE_MKL
                                            _GLIBCXX_USE_TBB_PAR_BACKEND=0)
#target_link_libraries(dr_shp INTERFACE range-v3 MKL::MKL_DPCPP)
target_link_libraries(dr_shp INTERFACE MKL::MKL_DPCPP)
# if (NOT CXX_FORMAT_SUPPORT)
  target_link_libraries(dr_shp INTERFACE fmt::fmt)
# endif()

if (DEFINED ONEDPL_USE_DR)
  target_compile_options(dr_shp INTERFACE "-DONEDPL_USE_DISTRIBUTED_RANGES" "-ftemplate-backtrace-limit=0")
endif()

# For use, see:
# https://github.com/illuhad/hipSYCL/blob/develop/doc/using-hipsycl.md#using-the-cmake-integration
# example: cmake .. -DhipSYCL_DIR=</hipsycl/install/lib/cmake/hipSYCL>
# -DHIPSYCL_TARGETS="<targets>"
if($(HIPSYCL_TARGETS))
  find_package(hipSYCL REQUIRED)
  add_sycl_to_target(TARGET dr_shp)
  target_compile_options(dr_shp INTERFACE --hipsycl-targets='cuda:sm_75')
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(
  shp-tests
  shp-tests.cpp 
  ../common/all.cpp 
  ../common/copy.cpp 
  ../common/counted.cpp
  ../common/distributed_vector.cpp 
  ../common/drop.cpp # test fail 
  ../common/enumerate.cpp
  ../common/fill.cpp 
  ../common/for_each.cpp
  ../common/iota.cpp
  # ../common/iota_view.cpp # compile fail, even in distributed-ranges project
  ../common/reduce.cpp 
  ../common/sort.cpp # test fail
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
  target_link_libraries(${test-exec} GTest::gtest_main DR::shp)
  # if (NOT CXX_FORMAT_SUPPORT)
    target_link_libraries(${test-exec} fmt::fmt)
  # endif()
endforeach()

add_custom_target(shp-all-tests)

function(add_shp_ctest test_name name)
  add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL SHP)
  add_dependencies(shp-all-tests ${name})
endfunction()

add_shp_ctest(shp-tests shp-tests)
add_shp_ctest(shp-tests-3 shp-tests --devicesCount 3)
add_shp_ctest(shp-tests-3-only shp-tests-3 --devicesCount 3)
