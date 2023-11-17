##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

set(VARS_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux/vars.template.sh")
set(OUTPUT_VARS "${CMAKE_BINARY_DIR}/vars.sh")

# Skip the library build if it was built previously or should be taken from the package
if (NOT _ONEDPL_PSTL_OFFLOAD_BINARY_PATH)
    message(STATUS "Adding PSTL offload library build stage")

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

    add_custom_command(TARGET pstloffload POST_BUILD COMMAND
                ${CMAKE_COMMAND}
                -DDPL_ROOT=${CMAKE_SOURCE_DIR}
                -DVARS_TEMPLATE=${VARS_TEMPLATE}
                -DOUTPUT_VARS=${OUTPUT_VARS}
                -DPSTL_OFFLOAD_BINARY_PATH=${CMAKE_CURRENT_BINARY_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_vars.cmake
            )
else()
    message(STATUS "Skip PSTL offload library build")
    message(STATUS "Generate vars for binaries in ${_ONEDPL_PSTL_OFFLOAD_BINARY_PATH}")

    # Setting parameters for generate_vars.cmake script
    set(DPLROOT ${CMAKE_SOURCE_DIR})
    set(PSTL_OFFLOAD_BINARY_PATH ${_ONEDPL_PSTL_OFFLOAD_BINARY_PATH})

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_vars.cmake)
endif()
