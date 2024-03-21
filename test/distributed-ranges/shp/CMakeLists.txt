# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
  GIT_TAG 5300fe3)
FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 0b0f7cf)
FetchContent_MakeAvailable(cpp-format)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.0.0)
FetchContent_MakeAvailable(cxxopts)

find_package(MKL REQUIRED)

# include(GoogleTest)

add_library(dr_shp INTERFACE)
add_library(experimental::dr::shp ALIAS dr_shp)

target_include_directories(dr_shp INTERFACE . vendor ../../../include)
target_compile_definitions(dr_shp INTERFACE USE_MKL
                                            _GLIBCXX_USE_TBB_PAR_BACKEND=0)
target_link_libraries(dr_shp INTERFACE range-v3 fmt::fmt MKL::MKL_DPCPP)

if (DEFINED ONEDPL_USE_DR)
  target_compile_options(dr_shp INTERFACE "-DONEDPL_USE_DISTRIBUTED_RANGES")
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
  shp-tests.cpp ../common/all.cpp ../common/copy.cpp ../common/counted.cpp
  ../common/distributed_vector.cpp ../common/drop.cpp ../common/enumerate.cpp
  ../common/fill.cpp ../common/for_each.cpp ../common/iota.cpp
  # ../common/iota_view.cpp
  ../common/reduce.cpp ../common/sort.cpp ../common/subrange.cpp
  ../common/take.cpp ../common/transform.cpp ../common/transform_view.cpp
  ../common/zip.cpp ../common/zip_local.cpp containers.cpp algorithms.cpp
  copy.cpp detail.cpp fill.cpp gemv.cpp transform.cpp)

add_executable(shp-tests-3 shp-tests.cpp containers-3.cpp copy-3.cpp)

# skeleton for rapid builds of individual tests, feel free to change this
# add_executable(shp-quick-test shp-tests.cpp ../common/transform.cpp)
# target_compile_definitions(shp-quick-test PRIVATE QUICK_TEST)

#foreach(test-exec IN ITEMS shp-tests shp-tests-3 shp-quick-test)
foreach(test-exec IN ITEMS shp-tests shp-tests-3)
  target_link_libraries(${test-exec} GTest::gtest_main DR::shp fmt::fmt cxxopts)
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
