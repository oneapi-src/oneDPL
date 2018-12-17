# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

cmake_minimum_required(VERSION 3.1)
project(gamma_correction VERSION 1.0.0 LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)

# Add an executable target to "gamma_correction" project from source files
add_executable(${PROJECT_NAME} gamma_correction.cpp utils.cpp)

# Instruct CMake to process CMakeLists.txt file located in Parallel STL root directory
# in order to find "ParallelSTL" project described there
if (NOT DEFINED PSTL_ROOT)
    set(PSTL_ROOT ../../)
endif()
add_subdirectory(${PSTL_ROOT} parallelstl)

# Integrate ParallelSTL artifacts into gamma_correction's executable target
target_link_libraries(${PROJECT_NAME}
                      ParallelSTL
                      $<$<CXX_COMPILER_ID:GNU>:rt>)

# Enable SIMD vectorization by passing an OpenMP SIMD flag to the compiler.
# For more information about Parallel STL and it's usage of SIMD vectorization, visit
# https://software.intel.com/en-us/get-started-with-pstl
#
# Since OpenMP SIMD flag is compiler-specific, we need to choose it
# depending on what compiler and Operating System are being used
string (JOIN "\n" CONFIGS_TO_SIMD_FLAGS_MAP
    # "CMAKE_CXX_COMPILER_ID-CMAKE_SYSTEM_NAME:<compiler flag for that configuration>"
    "Intel-Windows:/Qopenmp-simd"
    "Intel-ANY:-qopenmp-simd"
    "GNU-Linux:-fopenmp-simd"
    "AppleClang-Darwin:-openmp-simd"
    "ANY-ANY:"
)

set (MY_CONFIG_REGEX "(${CMAKE_CXX_COMPILER_ID}|ANY)-(${CMAKE_SYSTEM_NAME}|ANY):([^\n]*)")
string (REGEX MATCH ${MY_CONFIG_REGEX} MATCHED_MAP_ENTRY ${CONFIGS_TO_SIMD_FLAGS_MAP})
set (SIMD_FLAG ${CMAKE_MATCH_3})

# Add OpenMP SIMD flag only if it is supported by your compiler
if (NOT ${SIMD_FLAG} STREQUAL "")
    include(CheckCXXCompilerFlag)

    string(MAKE_C_IDENTIFIER ${SIMD_FLAG} FLAG_DISPLAY_NAME)
    check_cxx_compiler_flag(${SIMD_FLAG} COMPILER_SUPPORTS_${FLAG_DISPLAY_NAME})

    if (COMPILER_SUPPORTS_${FLAG_DISPLAY_NAME})
        target_compile_options(${PROJECT_NAME} PRIVATE ${SIMD_FLAG})
    endif()
endif()

# By default, Parallel STL uses TBB library as backend.
# TBB_USE_DEBUG macro controls certain debugging features.
# For more info, visit
# https://www.threadingbuildingblocks.org/docs/help/reference/environment/enabling_debugging_features.html
target_compile_definitions(${PROJECT_NAME} PRIVATE
                           $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
                           $<$<CONFIG:Release>:NDEBUG>)

