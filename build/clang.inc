##===-- clang.inc ---------------------------------------------------------===##
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

override compiler:=clang++

# XHOST_FLAG = -fno-vectorize
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
# XHOST_FLAG = $(KEY)mavx2 #-fno-vectorize
# XHOST_FLAG = $(KEY)mavx512f #-fno-vectorize
# DYN_LDFLAGS += $(LINK_KEY)c++
# CPLUS_FLAGS += -stdlib=libc++

# Match "clang version x.y.z" (not "x.y.z") to avoid matching on "Apple LLVM
# version x.y.z" which does not correspond upstream LLVM version.
ifneq (,$(shell clang --version | egrep  "clang version [6-9]\.[0-9]\.[0-9]"))
CPLUS_FLAGS += -fopenmp-simd #supported at least since 6.0 version
ifneq ($(target),android)
    PSTL_ARCH += $(KEY)march=native
endif
endif
