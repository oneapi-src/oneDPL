#!/bin/sh
##===-- onedpl_install.sh -------------------------------------------------===##
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
ONEDPL_INSTALL_DST=""
if [ -n "$1" ]; then
    ONEDPL_INSTALL_DST=$1
else
    echo "ERROR: please set oneDPL install dir"
    return 1;
fi

if [ -e $ONEDPL_INSTALL_DST/onedpl ]; then
    echo "onedpl already exists in '$ONEDPL_INSTALL_DST'"
    return 1;
fi

mkdir -p ${ONEDPL_INSTALL_DST}/onedpl
ONEDPL_SRC=$(cd $(dirname ${BASH_SOURCE}) && pwd -P)/..
cp -r ${ONEDPL_SRC}/include ${ONEDPL_INSTALL_DST}/onedpl
