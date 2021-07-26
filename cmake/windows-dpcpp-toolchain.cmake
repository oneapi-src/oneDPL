##===-- windows-dpcpp-toolchain.cmake -------------------------------------===##
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

if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG_INIT "/debug")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO_INIT "/debug")

if (NOT ${CMAKE_VERSION} VERSION_LESS "3.20")
	execute_process(COMMAND ${CMAKE_CXX_COMPILER} /clang:-dumpversion OUTPUT_VARIABLE COMPILER_VERSION)
	string(REGEX REPLACE "\n" "" COMPILER_VERSION "${COMPILER_VERSION}")
	set(CMAKE_CXX_COMPILER_ID "Clang ${COMPILER_VERSION}" CACHE STRING "Switch compiler identification" FORCE)
endif()

include(Platform/Windows-Clang)
set(CMAKE_LINKER ${CMAKE_CXX_COMPILER})
set(MSVC TRUE)
