@echo off
REM
REM ===----------------------------------------------------------------------===
REM
REM Copyright (C) Intel Corporation
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

if not defined SETVARS_CALL (
    echo:
    echo :: ERROR: This script must be executed by setvars.bat.
    echo:   Try '[install-dir]\setvars.bat --help' for help.
    echo:
    exit /b 255
)

if not defined ONEAPI_ROOT (
    echo:
    echo :: ERROR: This script requires that the ONEAPI_ROOT env variable is set."
    echo:   Try '[install-dir]\setvars.bat --help' for help.
    echo:
    exit /b 254
)

set "VARSDIR=%~dp0"

if not defined DPL_ROOT for /f "delims=" %%F in ("%VARSDIR%..") do set "DPL_ROOT=%%~fF"

set "CMAKE_PREFIX_PATH=%DPL_ROOT%\lib\cmake\oneDPL;%CMAKE_PREFIX_PATH%"
