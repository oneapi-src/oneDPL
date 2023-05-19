##===-- oneDPLWindowsIntelLLVMConfig.cmake --------------------------------------===##
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

if (CMAKE_HOST_WIN32)
    if (CMAKE_VERSION VERSION_LESS 3.20)
        set(REASON_FAILURE "oneDPLWindowsIntelLLVM requires CMake 3.20 or later on Windows.")
        set(oneDPLWindowsIntelLLVM_FOUND FALSE)
        return()
    else()
        # Requires version 3.20 for baseline support of icx, icx-cl
        cmake_minimum_required(VERSION 3.20)
    endif()
endif()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

#if this is included after project() and oneDPLWindowsIntelLLVM hasn't already been included (in a previous cmake call), we are too late to make the compiler changes we need to
if (DEFINED CMAKE_PROJECT_NAME AND (NOT DEFINED oneDPLWindowsIntelLLVM_DIR))
    set(oneDPLWindowsIntelLLVM_FOUND False)
    set(REASON_FAILURE "oneDPLWindowsIntelLLVM package must be included before the project() call!")
else()

    # CMAKE_CXX_COMPILER_ID and CMAKE_CXX_COMPILER_VERSION cannot be used because
    # CMake 3.19 and older will detect IntelLLVM compiler as CLang with CLang-specific version, see https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    if (CMAKE_CXX_COMPILER MATCHES ".*(dpcpp-cl|dpcpp|icx-cl|icpx|icx)(.exe)?$")
        set(INTEL_LLVM_COMPILER TRUE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTEL_LLVM_COMPILER_VERSION_RAW)
        string(REGEX MATCH "[0-9][0-9][0-9][0-9]\\.[0-9]\\.[0-9]" INTEL_LLVM_COMPILER_VERSION ${INTEL_LLVM_COMPILER_VERSION_RAW})
        if (CMAKE_CXX_COMPILER MATCHES ".*(dpcpp-cl|icx-cl|icx)(.exe)?$")
            set(INTEL_LLVM_COMPILER_MSVC_LIKE TRUE)
            set(MSVC TRUE)
        else()
            set(INTEL_LLVM_COMPILER_GNU_LIKE TRUE)
        endif()
    else()
        set(INTEL_LLVM_COMPILER FALSE)
    endif()

    if (CMAKE_HOST_WIN32 AND INTEL_LLVM_COMPILER_GNU_LIKE)
        set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0")
        set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -DNDEBUG")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT "-debug")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT "-debug")

        #CMake does not properly handle IntelLLVM compilers with GNU-like front ends after 3.19,
        if (NOT ${CMAKE_VERSION} VERSION_LESS "3.20")
            set(CMAKE_CXX_COMPILER_ID "IntelLLVM GNU-Like Workaround" CACHE STRING "Switch compiler identification" FORCE)
            set(CMAKE_CXX_COMPILER_VERSION "${INTEL_LLVM_COMPILER_VERSION}" CACHE STRING "Switch compiler version" FORCE)
            # Explicitly setting the c++17 standard compile option so that "target_compile_features()" check functions properly for this workaround
            set(CMAKE_CXX_COMPILE_FEATURES cxx_std_14 cxx_std_17 cxx_std_20)
            set(CMAKE_CXX14_COMPILE_FEATURES cxx_std_14)
            set(CMAKE_CXX17_COMPILE_FEATURES cxx_std_17)
            set(CMAKE_CXX20_COMPILE_FEATURES cxx_std_20)
            set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std=c++14")
            set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
            set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std=c++20")
            set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT 14)
            set(INTELLLVM_WIN_STD_IGNORE_FIX TRUE)
            message(WARNING "On Windows, ${CMAKE_CXX_COMPILER} is not supported by CMake (https://gitlab.kitware.com/cmake/cmake/-/issues/24314) at this time and may encounter issues. We recommend using CMAKE_CXX_COMPILER=icx on Windows.")
        endif()
    endif()

    if (CMAKE_HOST_WIN32 AND INTEL_LLVM_COMPILER_MSVC_LIKE AND (NOT ${CMAKE_VERSION} VERSION_LESS "3.20") AND (${CMAKE_VERSION} VERSION_LESS "3.23"))
        # Fixing linker rules to provide the linker options to the linker despite being before "/link" (see below)
        # Adapted from fix in CMake 3.23: https://github.com/Kitware/CMake/commit/5d5a7123034361b6cacff96d3ed20d2bb78c33cc
        set(CMAKE_EXE_LINKER_FLAGS_INIT "/Qoption,link,/machine:x64")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE_INIT "/Qoption,link,/INCREMENTAL:NO")
        set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT "/Qoption,link,/debug /Qoption,link,/INCREMENTAL:NO")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT "/Qoption,link,/debug /Qoption,link,/INCREMENTAL")
        set(CMAKE_CXX_CREATE_CONSOLE_EXE "/Qoption,link,/subsystem:console")
        set(CMAKE_CXX_CREATE_WINDOWS_EXE "/Qoption,link,/subsystem:windows")
        message(WARNING "${CMAKE_CXX_COMPILER} requires changes to linker settings to allow proper usage with CMake ${CMAKE_VERSION} on Windows.  A workaround is provided but may have limitations. We recommend using CMake version 3.23.0 or newer on Windows")
    endif()


    # The following are planned workarounds to be applied after project() by oneDPL CMake files
    if (CMAKE_HOST_WIN32 AND INTEL_LLVM_COMPILER_MSVC_LIKE)
        # Fix std compiler options for icx, icx-cl
        # Adapted from fix in CMake 3.26: https://github.com/Kitware/CMake/commit/42ca6416afeabd445bc6c19749e68604c9c2d733
        if (${CMAKE_VERSION} VERSION_LESS "3.26")
            set(INTELLLVM_MSVC_WIN_STDOPTION_FIX TRUE)
        endif()

        if ((NOT ${CMAKE_VERSION} VERSION_LESS "3.20") AND (${CMAKE_VERSION} VERSION_LESS "3.23"))
            # Fixing linker rule to use the compiler for linking and moving link options before /link
            # Adapted from fix in CMake 3.23: https://github.com/Kitware/CMake/commit/5d5a7123034361b6cacff96d3ed20d2bb78c33cc
            set(INTELLLVM_MSVC_WIN_LINKORDER_FIX TRUE)

            if (${CMAKE_VERSION} VERSION_LESS "3.21")
                # Fixing issue with version CMAKE_CXX_STANDARD being ignored in CMake 3.20 https://github.com/Kitware/CMake/commit/84036d30d4bae01ed94602ebce7a404300fd7e5f
                set(INTELLLVM_WIN_STD_IGNORE_FIX TRUE)
            endif()
        elseif((NOT ${CMAKE_VERSION} VERSION_LESS "3.23") AND (${CMAKE_VERSION} VERSION_LESS "3.25"))
            # Intel provided workaround for CMake version 3.23+
            set(INTELLLVM_WIN_OFFICIAL_SUPPORT_FIX TRUE)
        endif()
    endif()

    if (INTELLLVM_MSVC_WIN_STDOPTION_FIX OR INTELLLVM_MSVC_WIN_LINKORDER_FIX OR INTELLLVM_WIN_STD_IGNORE_FIX OR INTELLLVM_WIN_OFFICIAL_SUPPORT_FIX)
        # Set up oneDPLWindowsIntelLLVMApply.cmake to be code injected at the end of the 'project()' call for the cmake project using this
        # This is required for workarounds which must be applied after the project() call
        set(CMAKE_PROJECT_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/oneDPLWindowsIntelLLVMApply.cmake)
    endif()
endif()

find_package_handle_standard_args(oneDPLWindowsIntelLLVM
    FOUND_VAR oneDPLWindowsIntelLLVM_FOUND
    REQUIRED_VARS INTEL_LLVM_COMPILER
    REASON_FAILURE_MESSAGE "${REASON_FAILURE}")
