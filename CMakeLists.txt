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

project(oneDPL VERSION 2021.1 LANGUAGES CXX)

option(ONEDPL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
option(ONEDPL_USE_RANGES_API "Enable the use of ranges API for algorithms" OFF)
option(ONEDPL_USE_UNNAMED_LAMBDA "Pass -fsycl-unnamed-lambda compile option" OFF)
option(ONEDPL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
set(ONEDPL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)

###############################################################################
# Setup the oneDPL library target
###############################################################################
add_library(oneDPL INTERFACE)
add_library(oneAPI::DPL ALIAS oneDPL)

# Build type by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Parallel STL test macro
target_compile_definitions(oneDPL INTERFACE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)

if (ONEDPL_USE_PARALLEL_POLICIES)
    message(STATUS "Using Parallel Policies")
    if (ONEDPL_BACKEND MATCHES "^(tbb|sycl|sycl_only)$")
        string(TOUPPER "ONEDPL_BACKEND" ONEDPL_BACKEND_NAME)
        set(USE_${ONEDPL_BACKEND_NAME}_BACKEND TRUE)

        # prepare TBB
        find_package(TBB 2021 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
        message(STATUS "oneDPL uses oneTBB ${TBB_VERSION}")
        target_link_libraries(oneDPL INTERFACE TBB::tbb)
        target_compile_definitions(oneDPL INTERFACE
            $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
            $<$<CONFIG:Debug>:PSTL_USE_DEBUG>
            $<$<BOOL:${USE_SYCL_ONLY_BACKEND}>:ONEDPL_USE_TBB_BACKEND=0> 
            $<$<BOOL:${USE_TBB_BACKEND}>:ONEDPL_USE_DPCPP_BACKEND=0>
            )

        # check compiler in case with DPC++
        check_cxx_compiler_flag("-fsycl" CXX_COMPILER_OPTION_FSYCL)
        if(NOT CXX_COMPILER_OPTION_FSYCL)
            message(STATUS "Compiler doesn't support SYCL extensions")
        endif()

        # DPC++ specific settings
        if (CMAKE_CXX_COMPILER MATCHES "dpcpp")
            if (NOT ONEDPL_DEVICE_TYPE)
                set(ONEDPL_DEVICE_TYPE "GPU")
            endif()

            if (ONEDPL_DEVICE_TYPE MATCHES "^(CPU|GPU|FPGA_HW|FPGA_EMU?)$")
                message(STATUS "Compiling for device type: ${ONEDPL_DEVICE_TYPE}")
                set(USE_${ONEDPL_DEVICE_TYPE} TRUE)
            else()
                # TODO: print list of supported devices.
                message(FATAL_ERROR "Unsupported device type: ${ONEDPL_DEVICE_TYPE}")
            endif()

            # Check correctness of STATIC_REPORT
            if(ONEDPL_FPGA_STATIC_REPORT)
                set(ENABLE_STATIC_REPORT 1)
                if(NOT USE_FPGA_HW)
                    message(FATAL_ERROR "Static report can only be generated for FPGA hardware")
                else()
                    message(STATUS "Static report will be generated")
                endif()
            endif()

            # DPC++ specific compiler options
            target_compile_options(oneDPL INTERFACE
                $<$<BOOL:${ONEDPL_USE_UNNAMED_LAMBDA}>:-fsycl-unnamed-lambda>
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:-fintelfpga>
                )

            # DPC++ specific macro
            target_compile_definitions(oneDPL INTERFACE
                $<$<BOOL:${ONEDPL_USE_RANGES_API}>:_ONEDPL_USE_RANGES=1>
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:ONEDPL_FPGA_DEVICE>
                $<$<BOOL:${USE_FPGA_EMU}>:ONEDPL_FPGA_EMULATOR>
                )
            
            # DPC++ specific link options
            target_link_libraries(oneDPL INTERFACE
                $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>>:-Xshardware>
                # TODO: make correct option set for FPGA
                # $<$<OR:$<BOOL:${USE_FPGA_HW}>,$<BOOL:${USE_FPGA_EMU}>,$<NOT:${ENABLE_STATIC_REPORT}>>:-fsycl-link>
                )
        endif()

    else()
        message(STATUS "Using Parallel Policies, but not oneTBB/DPC++")
        if (TARGET ${ONEDPL_BACKEND})
            target_link_libraries(oneDPL INTERFACE ${ONEDPL_BACKEND})
        else()
            find_package(${ONEDPL_BACKEND} REQUIRED)
            target_link_libraries(oneDPL INTERFACE ${${ONEDPL_BACKEND}_IMPORTED_TARGETS})
        endif()
    endif()

else()
    target_compile_definitions(oneDPL INTERFACE
        PSTL_USE_PARALLEL_POLICIES=0
        ONEDPL_USE_TBB_BACKEND=0
        ONEDPL_USE_DPCPP_BACKEND=0
        )
endif()

target_include_directories(oneDPL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:stdlib>)

target_compile_features(oneDPL
    INTERFACE
    cxx_constexpr
    cxx_inline_namespaces)

###############################################################################
# Setup tests
###############################################################################
enable_testing()
add_subdirectory(test)
