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
  message(STATUS "using fmt library for logging in distributed-ranges tests")
# endif()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

# FetchContent_Declare(
#   range-v3
#   GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
#   GIT_TAG 5300fe3)
# FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.0.0)
FetchContent_MakeAvailable(cxxopts)

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
  # ../common/enumerate.cpp # compile fail
  ../common/fill.cpp 
  ../common/for_each.cpp # compile fail
  ../common/iota.cpp
  ## ../common/iota_view.cpp # compile fail
  ../common/reduce.cpp 
  ../common/sort.cpp # test fail
  ../common/subrange.cpp
  ../common/take.cpp 
  ../common/transform.cpp 
  ../common/transform_view.cpp
  ../common/zip.cpp # compile fail
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
  target_link_libraries(${test-exec} GTest::gtest_main DR::shp cxxopts)
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
