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

if (${CMAKE_VERSION} GREATER_EQUAL "3.20")
	string(REGEX REPLACE "dpcpp-cl" "dpcpp" DPCPP_COMPILER "${CMAKE_CXX_COMPILER}")
	execute_process(COMMAND ${DPCPP_COMPILER} -dumpversion OUTPUT_VARIABLE COMPILER_VERSION)
	string(REGEX REPLACE "\n" "" COMPILER_VERSION "${COMPILER_VERSION}")
	set(CMAKE_CXX_COMPILER_ID "Clang ${COMPILER_VERSION}" CACHE STRING "Switch compiler identification" FORCE)
endif()

include(Platform/Windows-Clang)
set(CMAKE_LINKER ${CMAKE_CXX_COMPILER})
set(MSVC TRUE)
