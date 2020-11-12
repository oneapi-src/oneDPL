##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) 2018-2019 Intel Corporation
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

cmake_minimum_required(VERSION 3.4.3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(PARALLELSTL_VERSION_FILE "include/oneapi/dpl/pstl/pstl_config.h")
file(STRINGS "${PARALLELSTL_VERSION_FILE}" PARALLELSTL_VERSION_SOURCE REGEX "#define __INTEL_PSTL_VERSION .*$")
string(REGEX REPLACE "#define __INTEL_PSTL_VERSION (.*)$" "\\1" PARALLELSTL_VERSION_SOURCE "${PARALLELSTL_VERSION_SOURCE}")
math(EXPR VERSION_MAJOR "${PARALLELSTL_VERSION_SOURCE} / 100")
math(EXPR VERSION_MINOR "${PARALLELSTL_VERSION_SOURCE} % 100")

project(ParallelSTL VERSION ${VERSION_MAJOR}.${VERSION_MINOR} LANGUAGES CXX)

option(PARALLELSTL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
option(PARALLELSTL_USE_RANGES_API "Enable the use of ranges API for algorithms" OFF)
option(PARALLELSTL_USE_UNNAMED_LAMBDA "Pass -fsycl-unnamed-lambda compile option" OFF)
option(PARALLELSTL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
set(PARALLELSTL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)

if (NOT TBB_DIR)
    get_filename_component(PSTL_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(REPLACE pstl tbb TBB_DIR_NAME ${PSTL_DIR_NAME})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${TBB_DIR_NAME}/cmake")
        get_filename_component(TBB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${TBB_DIR_NAME}/cmake" ABSOLUTE)
    endif()
endif()

###############################################################################
# Setup the ParallelSTL library target
###############################################################################
add_library(ParallelSTL INTERFACE)
add_library(pstl::ParallelSTL ALIAS ParallelSTL)

# Build type by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(PARALLELSTL_USE_DEBUG_MODE TRUE)
endif()

# Parallel STL test macro
target_compile_definitions(ParallelSTL INTERFACE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)

if (PARALLELSTL_USE_PARALLEL_POLICIES)
    message(STATUS "Using Parallel Policies")
    if (PARALLELSTL_BACKEND MATCHES "^(tbb|sycl)$")
        # prepare TBB
        find_package(TBB 2018 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
        message(STATUS "Parallel STL uses TBB ${TBB_VERSION} (interface version: ${TBB_INTERFACE_VERSION})")
        target_link_libraries(ParallelSTL INTERFACE TBB::tbb)
        target_compile_definitions(ParallelSTL INTERFACE    
            _PSTL_TEST_SUCCESSFUL_KEYWORD=1
            $<$<BOOL:${PARALLELSTL_USE_DEBUG_MODE}>:TBB_USE_DEBUG=1>
            $<$<BOOL:${PARALLELSTL_USE_DEBUG_MODE}>:PSTL_USE_DEBUG>
            )

        # check compiler in case with DPC++
        check_cxx_compiler_flag("-fsycl" CXX_COMPILER_OPTION_FSYCL)
        if(NOT CXX_COMPILER_OPTION_FSYCL)
            message(STATUS "Compiler doesn't support SYCL extensions")
        endif()

        # Check correctness of BENCH_DEVICE_TYPE
        if(NOT PARALLELSTL_DEVICE_TYPE)
            message(WARNING "Device type hasn't defined: the default compile/link options are used")
        elseif(PARALLELSTL_DEVICE_TYPE MATCHES "^(CPU|GPU|FPGA_HW|FPGA_EMU?)$")
            message(STATUS "Compile/link options for ${PARALLELSTL_DEVICE_TYPE} are used")
            set(USE_${PARALLELSTL_DEVICE_TYPE} TRUE)
        endif()

        # DPC++ specific settings
        if (CMAKE_CXX_COMPILER MATCHES "dpcpp")
            # Check correctness of STATIC_REPORT
            if(PARALLELSTL_FPGA_STATIC_REPORT)
                if(NOT USE_FPGA_HW)
                    message(FATAL_ERROR "Static report can only be generated for FPGA hardware")
                else()
                    message(STATUS "Static report will be generated")
                endif()
            endif()

            # DPC++ specific compiler options
            target_compile_options(ParallelSTL INTERFACE
                $<$<BOOL:${PARALLELSTL_USE_UNNAMED_LAMBDA}>:-fsycl-unnamed-lambda>
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:-fintelfpga>
                )

            # DPC++ specific macro
            target_compile_definitions(ParallelSTL INTERFACE
                $<$<BOOL:${PARALLELSTL_USE_RANGES_API}>:_ONEDPL_USE_RANGES=1>
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:ONEDPL_FPGA_DEVICE>
                $<$<BOOL:${USE_FPGA_EMU}>:ONEDPL_FPGA_EMULATOR>
                )
            
            # DPC++ specific link options
            target_link_libraries(ParallelSTL INTERFACE
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:-Xshardware>
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>,$<NOT:${PARALLELSTL_FPGA_STATIC_REPORT}>>:-fsycl-link>
                )
        endif()

    else()
        message(STATUS "Using Parallel Policies, but not tbb/dpc++")
        if (TARGET ${PARALLELSTL_BACKEND})
            target_link_libraries(ParallelSTL INTERFACE ${PARALLELSTL_BACKEND})
        else()
            find_package(${PARALLELSTL_BACKEND} REQUIRED)
            target_link_libraries(ParallelSTL INTERFACE ${${PARALLELSTL_BACKEND}_IMPORTED_TARGETS})
        endif()
    endif()

else()
    target_compile_definitions(ParallelSTL INTERFACE
        PSTL_USE_PARALLEL_POLICIES=0
        ONEDPL_USE_TBB_BACKEND=0
        ONEDPL_USE_DPCPP_BACKEND=0
        )
endif()

target_include_directories(ParallelSTL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:stdlib>)

target_compile_features(ParallelSTL
    INTERFACE
    cxx_constexpr
    cxx_inline_namespaces)

###############################################################################
# Setup tests
###############################################################################
enable_testing()
add_subdirectory(test)

###############################################################################
# Install the target and the associated CMake files
###############################################################################
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake"
                                 COMPATIBILITY ExactVersion)

configure_file(cmake/ParallelSTLConfig.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake"
               @ONLY)

install(TARGETS ParallelSTL
        EXPORT ParallelSTLTargets)
install(EXPORT ParallelSTLTargets
        FILE ParallelSTLTargets.cmake
        NAMESPACE pstl::
        DESTINATION lib/cmake/ParallelSTL)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake"
        DESTINATION lib/cmake/ParallelSTL)
install(DIRECTORY include/pstl
        DESTINATION include)

add_custom_target(install-pstl
                  COMMAND "${CMAKE_COMMAND}" -P "${PROJECT_BINARY_DIR}/cmake_install.cmake" -DCOMPONENT=ParallelSTL)
