##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/pstl_offload.cpp _pstl_offload_version_info)
string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_MAJOR ([0-9]+).*" "\\1" _pstl_offload_version_major "${_pstl_offload_version_info}")
string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_MINOR ([0-9]+).*" "\\1" _pstl_offload_version_minor "${_pstl_offload_version_info}")
string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_PATCH ([0-9]+).*" "\\1" _pstl_offload_version_patch "${_pstl_offload_version_info}")

add_library(pstloffload SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pstl_offload.cpp)

target_compile_options(pstloffload PRIVATE
            -fsycl
            -Wall -Wextra -Wformat -Wformat-security -Wremarks -Werror
            # have to put -fPIC after -fPIE to support linking when TLS is in use
            -fPIE -fPIC -mretpoline
            -fstack-protector
            )

target_link_options(pstloffload PRIVATE -Wl,-z,relro)

target_include_directories(pstloffload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties(pstloffload PROPERTIES
    VERSION ${_pstl_offload_version_major}.${_pstl_offload_version_minor}.${_pstl_offload_version_patch}
    SOVERSION ${_pstl_offload_version_major}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
