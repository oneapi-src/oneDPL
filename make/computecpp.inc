##===-- computecpp.inc ----------------------------------------------------===##
#
# Copyright (C) 2017-2020 Intel Corporation
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

override compiler:=compute++

CPLUS_FLAGS += -MT -no-serial-memop -sycl-driver
ifneq ($(stdver),)
    CPLUS_FLAGS += -std=$(stdver)
endif
DYN_LDFLAGS += -ltbb

ifeq ($(OS_NAME), windows)
	DYN_LDFLAGS += -lComputeCpp_vs2015
else
	DYN_LDFLAGS += -lComputeCpp
endif

