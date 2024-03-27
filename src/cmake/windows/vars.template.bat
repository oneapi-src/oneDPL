@echo off
REM
REM ===----------------------------------------------------------------------===
REM
REM Copyright (C) 2023 Intel Corporation
REM
REM SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
REM
REM This file incorporates work covered by the following copyright and permission
REM notice:
REM
REM Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
REM See https://llvm.org/LICENSE.txt for license information.
REM
REM ===----------------------------------------------------------------------===
REM

set SYCL_ENABLE_DEFAULT_CONTEXTS=1
set DPL_ROOT=@DPLROOT_REPLACEMENT@

set LIB=@LIBRARY_PATH_REPLACEMENT@;%LIB%
set PATH=@LIBRARY_PATH_REPLACEMENT@;%PATH%
