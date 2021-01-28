##===-- gcc.inc -----------------------------------------------------------===##
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

override compiler:=g++
XHOST_FLAG = $(KEY)march=native #-fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx2 #-fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx512f #-fno-tree-vectorize
 DYN_LDFLAGS += $(LINK_KEY)stdc++
# GCC 4.8.5 and early doesn't  support -fopenmp-simd; GCC 4.9 supports OpenMP 4.0 for C/C++
ifneq (,$(shell gcc -dumpversion | grep --invert-match "^[0-3]\.\|4\.[0-8]"))
    CPLUS_FLAGS += -fopenmp-simd
#   CPLUS_FLAGS += -fdump-rtl-loop2 #use this option to enable optimization report
endif

ifneq ($(stdver),)
    CPLUS_FLAGS += $(FQKEY)std=$(stdver)
endif
