##===-- clang.inc ---------------------------------------------------------===##
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

override compiler:=clang++

# XHOST_FLAG = -fno-vectorize
ifneq ($(stdver),)
    CPLUS_FLAGS += $(FQKEY)std=$(stdver)
endif
# XHOST_FLAG = $(KEY)mavx2 #-fno-vectorize
# XHOST_FLAG = $(KEY)mavx512f #-fno-vectorize
# DYN_LDFLAGS += $(LINK_KEY)c++
# CPLUS_FLAGS += -stdlib=libc++

# Match "clang version x.y.z" (not "x.y.z") to avoid matching on "Apple LLVM
# version x.y.z" which does not correspond upstream LLVM version.
ifneq (,$(shell clang --version | egrep  "clang version [6-9]\.[0-9]\.[0-9]"))
CPLUS_FLAGS += -fopenmp-simd #supported at least since 6.0 version
endif

ifneq (,$(shell clang --version | egrep  "DPC"))
CPLUS_FLAGS += -fopenmp-simd
endif

# In case of TBB backend and using Clang with libstdc++ the version of the library should be specify explicitly via TBB_USE_GLIBCXX_VERSION
ifeq ($(backend), tbb)
export gcc_version:=$(shell gcc -dumpfullversion -dumpversion)
ifneq (,$(gcc_version))
    # TODO: do not assume that GCC minor and patchlevel versions are always single-digit.
    CPLUS_FLAGS += $(KEY)DTBB_USE_GLIBCXX_VERSION=$(subst .,0,$(gcc_version))
endif
endif

