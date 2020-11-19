##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) 2018-2020 Intel Corporation
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

set(ONEDPL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

option(ONEDPL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
option(ONEDPL_USE_RANGES_API "Enable the use of ranges API for algorithms" OFF)
option(ONEDPL_USE_UNNAMED_LAMBDA "Pass -fsycl-unnamed-lambda compile option" OFF)
option(ONEDPL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
option(ONEDPL_USE_OFFLINE_COMPILATION "Enable the offline compilation via OCLOC compiler" OFF)

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)

###############################################################################
# Setup the oneDPL library target
###############################################################################
add_library(oneDPL INTERFACE)
add_library(oneAPI::DPL ALIAS oneDPL)

# Build type by default
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Parallel STL test macro
target_compile_definitions(oneDPL INTERFACE _PSTL_TEST_SUCCESSFUL_KEYWORD=1)

if (ONEDPL_USE_PARALLEL_POLICIES)
    message(STATUS "Using Parallel Policies")
    string(TOLOWER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
    if (ONEDPL_BACKEND MATCHES "^(tbb|sycl|sycl_only)$")
        string(TOUPPER "ONEDPL_BACKEND" ONEDPL_BACKEND_NAME)
        set(ONEDPL_USE_BACKEND_${ONEDPL_BACKEND_NAME} TRUE)

        if (ONEDPL_BACKEND MATCHES "^(tbb|sycl)$")
            find_package(TBB 2021 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
            message(STATUS "oneDPL uses oneTBB ${TBB_VERSION}")
            target_link_libraries(oneDPL INTERFACE TBB::tbb)
        endif()

        target_compile_definitions(oneDPL INTERFACE
            $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
            $<$<CONFIG:Debug>:PSTL_USE_DEBUG>
            $<$<BOOL:${ONEDPL_USE_BACKEND_SYCL_ONLY}>:ONEDPL_USE_TBB_BACKEND=0> 
            $<$<BOOL:${ONEDPL_USE_BACKEND_TBB}>:ONEDPL_USE_DPCPP_BACKEND=0>
            )

        if (ONEDPL_BACKEND MATCHES "^(sycl|sycl_only)$")
            check_cxx_compiler_flag("-fsycl" _fsycl_option)
            if (NOT _fsycl_option)
                message(FATAL_ERROR "Compiler doesn't support SYCL extensions (-fsycl option).\n"
                "It is required if ONEDPL_BACKEND=${ONEDPL_BACKEND}")
            endif()

            # settings for the specific compilation type
            if (NOT ONEDPL_USE_OFFLINE_COMPILATION)
                message(STATUS "Use online compilation. No additional parameters needed")
            else()
                if (NOT ONEDPL_COMPILE_ARCH)
                    set(ONEDPL_COMPILE_ARCH "*")
                    message(STATUS "Offline compilation for all available architectures")
                else()
                    string(TOLOWER ${ONEDPL_COMPILE_ARCH} ONEDPL_COMPILE_ARCH)
                    message(STATUS "Offline compilation for ${ONEDPL_COMPILE_ARCH} architecture")
                endif()
                set(ONEDPL_OFFLINE_COMP_OPTION_REL "-Xs \"-device ${ONEDPL_COMPILE_ARCH}\"")
                set(ONEDPL_OFFLINE_COMP_OPTION_DEB "-Xs \"-device ${ONEDPL_COMPILE_ARCH} -internal_options -cl-kernel-debug-enable -options -cl-opt-disable\"")
                set(ONEDPL_OFFLINE_COMP_OPTION_RELDEB "-Xs \"-device ${ONEDPL_COMPILE_ARCH} -internal_options -cl-kernel-debug-enable\"")
            endif()

            # check device type
            if (NOT ONEDPL_DEVICE_TYPE)
                set(ONEDPL_DEVICE_TYPE "GPU")
                message(STATUS "Use GPU as default device")
            endif()

            string(TOUPPER ${ONEDPL_DEVICE_TYPE} ONEDPL_DEVICE_TYPE)
            if (ONEDPL_DEVICE_TYPE MATCHES "^(CPU|GPU|FPGA_EMU|FPGA_HW?)$")
                message(STATUS "Compilation for ${ONEDPL_DEVICE_TYPE}")
                set(ONEDPL_USE_DEVICE_${ONEDPL_DEVICE_TYPE} TRUE)
            else()
                message(FATAL_ERROR "Unsupported device type: ${ONEDPL_DEVICE_TYPE}.\n"
                    "Select one of the following devices: CPU, GPU, FPGA_EMU or FPGA_HW")
            endif()

            # Check correctness of STATIC_REPORT
            if (ONEDPL_FPGA_STATIC_REPORT)
                if (NOT ONEDPL_USE_DEVICE_FPGA_HW)
                    message(FATAL_ERROR "Static report can only be generated for FPGA hardware")
                else()
                    message(STATUS "Static report will be generated")
                endif()
            endif()

            # DPC++ specific compiler options
            target_compile_options(oneDPL INTERFACE
                $<$<BOOL:${ONEDPL_USE_UNNAMED_LAMBDA}>:-fsycl-unnamed-lambda>
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${OONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga> 
                )

            # DPC++ specific macro
            target_compile_definitions(oneDPL INTERFACE
                $<$<BOOL:${ONEDPL_USE_RANGES_API}>:_ONEDPL_USE_RANGES=1>
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${OONEDPL_USE_DEVICE_FPGA_EMU}>>:ONEDPL_FPGA_DEVICE>
                $<$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>:ONEDPL_FPGA_EMULATOR>
                )
            
            # DPC++ specific link options
            target_link_libraries(oneDPL INTERFACE
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${OONEDPL_USE_DEVICE_FPGA_EMU}>>:-Xshardware>
                $<$<AND:$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${OONEDPL_USE_DEVICE_FPGA_EMU}>>,$<BOOL:${ONEDPL_FPGA_STATIC_REPORT}>>:-fsycl-link>
                $<$<BOOL:${ONEDPL_USE_OFFLINE_COMPILATION}>:-fsycl-targets=spir64_gen-unknown-unknown-sycldevice>
                $<$<AND:$<CONFIG:Release>,$<BOOL:${ONEDPL_USE_OFFLINE_COMPILATION}>>:${ONEDPL_OFFLINE_COMP_OPTION_REL}>
                $<$<AND:$<CONFIG:Debug>,$<BOOL:${ONEDPL_USE_OFFLINE_COMPILATION}>>:${ONEDPL_OFFLINE_COMP_OPTION_DEB}>
                $<$<AND:$<CONFIG:RelWithDebInfo>,$<BOOL:${ONEDPL_USE_OFFLINE_COMPILATION}>>:${ONEDPL_OFFLINE_COMP_OPTION_RELDEB}>
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
    $<INSTALL_INTERFACE:include>)

target_compile_features(oneDPL
    INTERFACE
    cxx_constexpr
    cxx_inline_namespaces)

###############################################################################
# Setup tests
###############################################################################
enable_testing()
add_subdirectory(test)
