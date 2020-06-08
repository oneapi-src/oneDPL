#!/bin/sh
##===-- onedplvars.sh -----------------------------------------------------===##
#
# Copyright (C) 2019-2020 Intel Corporation
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
ONEDPLROOT=SUBSTITUTE_INSTALL_DIR_HERE
if [ -n "$1" ]; then
    ONEDPLROOT=$1
else
    echo "ERROR: please set oneDPL install dir"
    return 1;
fi
if [ -z "${CPATH}" ]; then
    CPATH="${ONEDPLROOT}/onedpl/include"; export CPATH
else
    CPATH="${ONEDPLROOT}/onedpl/include:$CPATH"; export CPATH
fi
