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

set(PARALLELSTL_VERSION_FILE "include/pstl/internal/pstl_config.h")
file(STRINGS "${PARALLELSTL_VERSION_FILE}" PARALLELSTL_VERSION_SOURCE REGEX "#define PSTL_VERSION .*$")
string(REGEX MATCH "#define PSTL_VERSION (.*)$" PARALLELSTL_VERSION_SOURCE "${PARALLELSTL_VERSION_SOURCE}")
math(EXPR VERSION_MAJOR "${PARALLELSTL_VERSION_SOURCE} / 100")
math(EXPR VERSION_MINOR "${PARALLELSTL_VERSION_SOURCE} % 100")

project(ParallelSTL VERSION ${VERSION_MAJOR}.${VERSION_MINOR} LANGUAGES CXX)

option(PARALLELSTL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
set(PARALLELSTL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

include(CMakePackageConfigHelpers)

add_library(ParallelSTL INTERFACE)
add_library(pstl::ParallelSTL ALIAS ParallelSTL)

if (PARALLELSTL_USE_PARALLEL_POLICIES)
    if (PARALLELSTL_BACKEND STREQUAL "tbb")
        find_package(TBB 2018 REQUIRED tbb)
        message(STATUS "Parallel STL uses TBB ${TBB_VERSION} (interface version: ${TBB_INTERFACE_VERSION})")
        target_link_libraries(ParallelSTL INTERFACE TBB::tbb)
    else()
        if (TARGET ${PARALLELSTL_BACKEND})
            target_link_libraries(ParallelSTL INTERFACE ${PARALLELSTL_BACKEND})
        else()
            find_package(${PARALLELSTL_BACKEND} REQUIRED)
            target_link_libraries(ParallelSTL INTERFACE ${${PARALLELSTL_BACKEND}_IMPORTED_TARGETS})
        endif()
    endif()
else()
    target_add_definitions(ParallelSTL INTERFACE PSTL_USE_PARALLEL_POLICIES=0)
endif()

target_include_directories(ParallelSTL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_file(
    ParallelSTLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake
    @ONLY)

export(TARGETS ParallelSTL NAMESPACE pstl:: FILE ParallelSTLTargets.cmake)
export(PACKAGE ParallelSTL)
