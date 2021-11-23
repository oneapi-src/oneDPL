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

find_program(FIND_GXX_EXE g++)
if (FIND_GXX_EXE)
    execute_process(COMMAND ${FIND_GXX_EXE} -dumpfullversion OUTPUT_VARIABLE _onedpl_gxx_version)
endif()

option(ONEDPL_FPGA_STATIC_REPORT "Enable the static report generation for the FPGA device" OFF)
option(ONEDPL_USE_AOT_COMPILATION "Enable the ahead of time compilation via OCLOC compiler" OFF)
option(ONEDPL_ENABLE_SIMD "Enable SIMD vectorization by passing an OpenMP SIMD flag to the compiler if supported" ON)

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

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
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

string(TOUPPER ${ONEDPL_BACKEND} ONEDPL_BACKEND)
message(STATUS "Using parallel policies with ${ONEDPL_BACKEND} backend")
string(TOLOWER ${ONEDPL_BACKEND} ONEDPL_BACKEND)

if (ONEDPL_ENABLE_SIMD)
    if (CMAKE_CXX_COMPILER MATCHES ".*dpcpp(-cl)?(.exe)?$")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE VER)
        string(REGEX MATCH "[0-9][0-9][0-9][0-9]\\.[0-9]\\.[0-9]" DPCPP_VER ${VER})
    endif()
    if (NOT CMAKE_CXX_COMPILER MATCHES ".*dpcpp(-cl)?(.exe)?$" OR NOT DPCPP_VER VERSION_LESS 2021.4)
        foreach(_simd_flag -fopenmp-simd /Qopenmp-simd -qopenmp-simd -openmp-simd)
            string(MAKE_C_IDENTIFIER ${_simd_flag} FLAG_DISPLAY_NAME)
            check_cxx_compiler_flag(${_simd_flag} ${FLAG_DISPLAY_NAME}_option)
            if (${FLAG_DISPLAY_NAME}_option)
                target_compile_options(oneDPL INTERFACE ${_simd_flag})
                set(_simd_enabled_flag ${_simd_flag})
                break()
            endif()
        endforeach()
    endif()

    if (_simd_enabled_flag)
        message(STATUS "oneDPL: OpenMP SIMD is enabled by passing '${_simd_enabled_flag}' to compiler")
    else()
        message(STATUS "oneDPL: no effect from enabled ONEDPL_ENABLE_SIMD; unsupported for current compiler")
    endif()
else()
    message(STATUS "oneDPL: ONEDPL_ENABLE_SIMD is OFF, corresponding compiler flag is not used")
endif()

if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp|dpcpp_only)$")
    string(TOUPPER "${ONEDPL_BACKEND}" ONEDPL_BACKEND_NAME)
    set(ONEDPL_USE_BACKEND_${ONEDPL_BACKEND_NAME} TRUE)

    if (ONEDPL_BACKEND MATCHES "^(tbb|dpcpp)$")
        find_package(TBB 2021 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
        message(STATUS "oneDPL uses oneTBB ${TBB_VERSION}")
        target_link_libraries(oneDPL INTERFACE TBB::tbb)
    endif()

    # It is for Clang and Intel® oneAPI DPC++ Compiler (while the last one is detected as Clang; for Linux only), which are used with libstdc++ standard library
    if (UNIX)
        if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
            if (FIND_GXX_EXE)
                string(REPLACE "\." "0" _onedpl_tbb_use_glibcxx_version ${_onedpl_gxx_version})
                target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=${_onedpl_tbb_use_glibcxx_version})
            else()
                target_compile_definitions(oneDPL INTERFACE TBB_USE_GLIBCXX_VERSION=70300)
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

        # check device backend
        if (ONEDPL_DEVICE_BACKEND MATCHES "^(opencl|level_zero)$")
            message(STATUS "Compilation using ${ONEDPL_DEVICE_BACKEND} device backend")
        elseif (DEFINED ONEDPL_DEVICE_BACKEND)
            message(FATAL_ERROR "Unsupported device backend: ${ONEDPL_DEVICE_BACKEND}.\n"
                "Select one of the following device backends: opencl or level_zero")
        else()
            set(ONEDPL_DEVICE_BACKEND "*")
            message(STATUS "Use a default device backend")
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
            $<$<OR:$<BOOL:${ONEDPL_USE_DEVICE_FPGA_HW}>,$<BOOL:${ONEDPL_USE_DEVICE_FPGA_EMU}>>:-fintelfpga>
            )
        if (DEFINED ONEDPL_USE_UNNAMED_LAMBDA)
            if(ONEDPL_USE_UNNAMED_LAMBDA)
                message(STATUS "Use unnamed lambdas")
                target_compile_options(oneDPL INTERFACE -fsycl-unnamed-lambda)
            else()
                message(STATUS "Don't use unnamed lambdas")
                target_compile_options(oneDPL INTERFACE -fno-sycl-unnamed-lambda)
            endif()
        endif()

        # DPC++ specific macro
        target_compile_definitions(oneDPL INTERFACE
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

elseif(ONEDPL_BACKEND MATCHES "^(serial)$")
    target_compile_definitions(oneDPL INTERFACE
        ONEDPL_USE_TBB_BACKEND=0
        ONEDPL_USE_DPCPP_BACKEND=0
        ONEDPL_USE_OPENMP_BACKEND=0
        )
    message(STATUS "Compilation for the host due to serial backend")

elseif(ONEDPL_BACKEND MATCHES "^(omp)$")
    if(UNIX)
        set(_openmp_flag "-fopenmp")
    else()
        set(_openmp_flag "-Qopenmp")
    endif()
    set(_cmake_required_libraries ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_LIBRARIES ${_openmp_flag})
    check_cxx_compiler_flag(${_openmp_flag} _openmp_option)
    if (_openmp_option)
        target_compile_options(oneDPL INTERFACE ${_openmp_flag})
        target_link_libraries(oneDPL INTERFACE ${_openmp_flag})
        target_compile_definitions(oneDPL INTERFACE
            ONEDPL_USE_TBB_BACKEND=0
            ONEDPL_USE_DPCPP_BACKEND=0
            ONEDPL_USE_OPENMP_BACKEND=1
            )
        message(STATUS "Compilation for the host due to OpenMP backend")
    else()
        message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support ${_openmp_flag} option.\n"
            "It is required if ONEDPL_BACKEND=${ONEDPL_BACKEND}")
    endif()
    set(CMAKE_REQUIRED_LIBRARIES ${_cmake_required_libraries})

else()
    message(STATUS "Using Parallel Policies, but not oneTBB/DPC++")
    if (TARGET ${ONEDPL_BACKEND})
        target_link_libraries(oneDPL INTERFACE ${ONEDPL_BACKEND})
    else()
        find_package(${ONEDPL_BACKEND} REQUIRED)
        target_link_libraries(oneDPL INTERFACE ${${ONEDPL_BACKEND}_IMPORTED_TARGETS})
    endif()
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

###############################################################################
# Installation instructions
###############################################################################
install(CODE "set(OUTPUT_DIR \"${CMAKE_INSTALL_FULL_LIBDIR}/cmake/oneDPL\")")
install(CODE "set(SKIP_HEADERS_SUBDIR TRUE)")
install(SCRIPT cmake/scripts/generate_config.cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY licensing DESTINATION ${CMAKE_INSTALL_DOCDIR})
