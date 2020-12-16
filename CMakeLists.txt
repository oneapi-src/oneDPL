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

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/oneapi/dpl/pstl/onedpl_config.h
    _onedpl_version_info
    LIMIT 1024)
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MAJOR ([0-9]+).*" "\\1" _onedpl_ver_major "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_MINOR ([0-9]+).*" "\\1" _onedpl_ver_minor "${_onedpl_version_info}")
string(REGEX REPLACE ".*#define ONEDPL_VERSION_PATCH ([0-9]+).*" "\\1" _onedpl_ver_patch "${_onedpl_version_info}")

project(oneDPL VERSION ${_onedpl_ver_major}.${_onedpl_ver_minor}.${_onedpl_ver_patch} LANGUAGES CXX)
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

option(ONEDPL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
option(ONEDPL_USE_RANGES_API "Enable the use of ranges API for algorithms" OFF)
option(ONEDPL_USE_UNNAMED_LAMBDA "Pass -fsycl-unnamed-lambda compile option" OFF)
option(ONEDPL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
option(ONEDPL_USE_AOT_COMPILATION "Enable the ahead of time compilation via OCLOC compiler" OFF)

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)

# Set default back-end in according with compiler (DPC++ or others)
check_cxx_compiler_flag("-fsycl" _fsycl_option)
if (NOT ONEDPL_BACKEND)
    if (_fsycl_option)
        set(ONEDPL_BACKEND "dpcpp" CACHE STRING "Threading backend")
    else()
        set(ONEDPL_BACKEND "tbb" CACHE STRING "Threading backend")
    endif()
    string(TOUPPER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
    message(STATUS "Use ${ONEDPL_BACKEND} as default backend")
endif()

###############################################################################
# Setup the oneDPL library target
###############################################################################
add_library(oneDPL INTERFACE)

if (CMAKE_BUILD_TYPE)
    message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type is not set")
endif()

if (MSVC)
    target_compile_options(oneDPL INTERFACE /Zc:__cplusplus /EHsc)
else()
    target_compile_options(oneDPL INTERFACE $<$<CONFIG:Debug>:-O0>)
endif()

if (ONEDPL_USE_PARALLEL_POLICIES)
    string(TOUPPER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
    message(STATUS "Using parallel policies with ${ONEDPL_BACKEND} backend")
    string(TOLOWER ${ONEDPL_BACKEND} ONEDPL_BACKEND)

    if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp|dpcpp_only)$")
        string(TOUPPER "${ONEDPL_BACKEND}" ONEDPL_BACKEND_NAME)
        set(ONEDPL_USE_BACKEND_${ONEDPL_BACKEND_NAME} TRUE)

        if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp)$")
            find_package(TBB 2021 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
            message(STATUS "oneDPL uses oneTBB ${TBB_VERSION}")
            target_link_libraries(oneDPL INTERFACE TBB::tbb)
        endif()

        # It is for Clang and DPC++ compilers (while the last one is detected as Clang; for Linux only), which are used with libstdc++ standard library
        if (UNIX)
            if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
                find_program(FIND_GXX_EXE g++)
                if (FIND_GXX_EXE)
                    execute_process(COMMAND ${FIND_GXX_EXE} -dumpfullversion OUTPUT_VARIABLE _onedpl_gxx_version)
                    string(REPLACE "\." "0" _onedpl_gxx_version ${_onedpl_gxx_version})
                    target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=${_onedpl_gxx_version})
                else()
                    target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=70300)
                endif()
            endif()
            if(CMAKE_CXX_STANDARD GREATER_EQUAL 17)
                if(_onedpl_gxx_version AND _onedpl_gxx_version GREATER_EQUAL 80100 OR NOT _onedpl_gxx_version)
                    set(ONEDPL_USE_RANGES_API ON)
                endif()
            endif()
        endif()

        target_compile_definitions(oneDPL INTERFACE
            $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
            $<$<CONFIG:Debug>:PSTL_USE_DEBUG>
            $<$<BOOL:${ONEDPL_USE_BACKEND_DPCPP_ONLY}>:ONEDPL_USE_TBB_BACKEND=0>
            $<$<BOOL:${ONEDPL_USE_BACKEND_TBB}>:ONEDPL_USE_DPCPP_BACKEND=0>
            )

        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            if (NOT _fsycl_option)
                message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support -fsycl option.\n"
                "It is required if ONEDPL_BACKEND=${ONEDPL_BACKEND}")
            endif()

            # settings for the specific compilation type
            if (NOT ONEDPL_USE_AOT_COMPILATION)
                message(STATUS "Use the ahead of time compilation. No additional parameters needed")
            else()
                if (NOT ONEDPL_AOT_ARCH)
                    set(ONEDPL_AOT_ARCH "*")
                    message(STATUS "Ahead of time compilation for all available architectures")
                else()
                    string(TOLOWER ${ONEDPL_AOT_ARCH} ONEDPL_AOT_ARCH)
                    message(STATUS "Ahead of time compilation for ${ONEDPL_AOT_ARCH} architecture")
                endif()
                set(ONEDPL_AOT_COMP_OPTION_REL "-Xs \"-device ${ONEDPL_AOT_ARCH}\"")
                set(ONEDPL_AOT_COMP_OPTION_DEB "-Xs \"-device ${ONEDPL_AOT_ARCH} -internal_options -cl-kernel-debug-enable -options -cl-opt-disable\"")
                set(ONEDPL_AOT_COMP_OPTION_RELDEB "-Xs \"-device ${ONEDPL_AOT_ARCH} -internal_options -cl-kernel-debug-enable\"")
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
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga>
                )

            # DPC++ specific macro
            target_compile_definitions(oneDPL INTERFACE
                $<$<BOOL:${ONEDPL_USE_RANGES_API}>:_ONEDPL_USE_RANGES=1>
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:ONEDPL_FPGA_DEVICE>
                $<$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>:ONEDPL_FPGA_EMULATOR>
                )

            # DPC++ specific link options
            target_link_libraries(oneDPL INTERFACE
                $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga>
                $<$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>:-Xshardware>
                $<$<AND:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_FPGA_STATIC_REPORT}>>:-fsycl-link>
                $<$<BOOL:${ONEDPL_USE_AOT_COMPILATION}>:-fsycl-targets=spir64_gen-unknown-unknown-sycldevice>
                $<$<AND:$<CONFIG:Release>,$<BOOL:${ONEDPL_USE_AOT_COMPILATION}>>:${ONEDPL_AOT_COMP_OPTION_REL}>
                $<$<AND:$<CONFIG:Debug>,$<BOOL:${ONEDPL_USE_AOT_COMPILATION}>>:${ONEDPL_AOT_COMP_OPTION_DEB}>
                $<$<AND:$<CONFIG:RelWithDebInfo>,$<BOOL:${ONEDPL_USE_AOT_COMPILATION}>>:${ONEDPL_AOT_COMP_OPTION_RELDEB}>
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

###############################################################################
# Setup tests
###############################################################################
enable_testing()
add_subdirectory(test)
