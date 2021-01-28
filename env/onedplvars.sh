#!/bin/sh
##===-- onedplvars.sh -----------------------------------------------------===##
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
if [ -n "$1" ]; then
    ONEDPLROOT=$1
else
    ONEDPLROOT=$(dirname ${BASH_SOURCE})/..
fi
if [ -z "${CPATH}" ]; then
    CPATH="${ONEDPLROOT}/include"; export CPATH
else
    CPATH="${ONEDPLROOT}/include:$CPATH"; export CPATH
fi
