##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

if (CMAKE_HOST_WIN32)
    set(VARS_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/windows/vars.template.bat")
    set(OUTPUT_VARS "${CMAKE_BINARY_DIR}/vars.bat")
else()
    set(VARS_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux/vars.template.sh")
    set(OUTPUT_VARS "${CMAKE_BINARY_DIR}/vars.sh")
endif()

# Skip the library build if it was built previously or should be taken from the package
if (NOT _ONEDPL_PSTL_OFFLOAD_BINARY_PATH)
    message(STATUS "Adding PSTL offload library build stage")

    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/pstl_offload.cpp _pstl_offload_version_info)
    string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_MAJOR ([0-9]+).*" "\\1" _pstl_offload_version_major "${_pstl_offload_version_info}")
    string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_MINOR ([0-9]+).*" "\\1" _pstl_offload_version_minor "${_pstl_offload_version_info}")
    string(REGEX REPLACE ".*#define _PSTL_OFFLOAD_BINARY_VERSION_PATCH ([0-9]+).*" "\\1" _pstl_offload_version_patch "${_pstl_offload_version_info}")

    add_library(pstloffload SHARED ${CMAKE_CURRENT_SOURCE_DIR}/pstl_offload.cpp)

    if (CMAKE_HOST_WIN32)
        set(EXTRA_COMPILE_FLAGS "/GS")
    else()
        # have to put -fPIC after -fPIE to support linking when TLS is in use
        set(EXTRA_COMPILE_FLAGS -fPIE -fPIC -fstack-protector -Wremarks)
    endif()

    target_compile_options(pstloffload PRIVATE
                -fsycl
                -Wall -Wextra -Wformat -Wformat-security -Werror
                -mretpoline
                ${EXTRA_COMPILE_FLAGS}
                )

    target_link_options(pstloffload PRIVATE -Wl,-z,relro)

    target_include_directories(pstloffload PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

    set_target_properties(pstloffload PROPERTIES
        VERSION ${_pstl_offload_version_major}.${_pstl_offload_version_minor}.${_pstl_offload_version_patch}
        SOVERSION ${_pstl_offload_version_major}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    add_custom_command(OUTPUT ${OUTPUT_VARS} COMMAND
        ${CMAKE_COMMAND}
        -DDPL_ROOT=${CMAKE_SOURCE_DIR}
        -DVARS_TEMPLATE=${VARS_TEMPLATE}
        -DOUTPUT_VARS=${OUTPUT_VARS}
        -DPSTL_OFFLOAD_BINARY_PATH=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_vars.cmake
    )
    add_custom_target(set_env_vars DEPENDS ${OUTPUT_VARS})
    add_dependencies(pstloffload set_env_vars)

    if (CMAKE_HOST_WIN32)
        include(ExternalProject)
        find_program(NMAKE_EXE NAMES nmake REQUIRED)
        set(DETOURS_PATH ${PROJECT_BINARY_DIR}/src/project_detours-prefix/src/project_detours)
        ExternalProject_Add(project_detours
            GIT_REPOSITORY    https://github.com/microsoft/Detours.git
            GIT_TAG           4b8c659f549b0ab21cf649377c7a84eb708f5e68
            INSTALL_COMMAND   ""
            CONFIGURE_COMMAND ""
            # use CL to add additional flags to the Detours build
            BUILD_COMMAND cd src && SET CL=/sdl
            COMMAND ${NMAKE_EXE}
            BUILD_IN_SOURCE on
            STEP_TARGETS      build
            BUILD_BYPRODUCTS ${DETOURS_PATH}/lib.X64/detours.lib
        )

        message(STATUS "Using Detours from ${DETOURS_PATH}")

        add_library(detours STATIC IMPORTED)
        set_property(TARGET detours PROPERTY IMPORTED_LOCATION ${DETOURS_PATH}/lib.X64/detours.lib)
        add_dependencies(detours project_detours)

        target_include_directories(pstloffload PRIVATE ${DETOURS_PATH}/include)
        target_link_libraries(pstloffload PRIVATE detours)

        set_target_properties(pstloffload PROPERTIES
            LINK_FLAGS "${CMAKE_LINK_DEF_FILE_FLAG}\"${CMAKE_CURRENT_SOURCE_DIR}/win64.def\""
            LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/win64.def"
        )
    endif()
else()
    message(STATUS "Skip PSTL offload library build")
    message(STATUS "Generate vars for binaries in ${_ONEDPL_PSTL_OFFLOAD_BINARY_PATH}")

    # Setting parameters for generate_vars.cmake script
    set(DPL_ROOT ${CMAKE_SOURCE_DIR})
    set(PSTL_OFFLOAD_BINARY_PATH ${_ONEDPL_PSTL_OFFLOAD_BINARY_PATH})

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_vars.cmake)
endif()
