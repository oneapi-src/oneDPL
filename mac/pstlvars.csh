#!/bin/csh
##===-- pstlvars.csh ------------------------------------------------------===##
#
# Copyright (C) 2017-2019 Intel Corporation
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

if ("$1" == "auto_pstlroot") then
    set sourced=($_)
    if ("$sourced" != '') then # if the script was sourced
        set script_name=$sourced[2]
    else # if the script was run => "$_" is empty
        set script_name=$0
    endif
    set dir_name=`dirname $script_name`
    set script_dir=`cd "$dir_name"; pwd -P`
    setenv PSTLROOT "$script_dir/.."
else
    setenv PSTLROOT "SUBSTITUTE_INSTALL_DIR_HERE"
endif

if ( -e $PSTLROOT/../tbb/bin/tbbvars.csh ) then
   source $PSTLROOT/../tbb/bin/tbbvars.csh;
endif

if (! $?CPATH) then
    setenv CPATH "${PSTLROOT}/include:${PSTLROOT}/stdlib"
else
    setenv CPATH "${PSTLROOT}/include:${PSTLROOT}/stdlib:$CPATH"
endif
