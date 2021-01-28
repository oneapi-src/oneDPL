##===-- icc.inc -----------------------------------------------------------===##
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

vecreport ?= 0#may be set to [0..7], see https://software.intel.com/en-us/node/522949 for details

XHOST_FLAG = $(KEY)xHOST #-no-vec #-qopt-zmm-usage=high
CPLUS_FLAGS += $(QKEY)opt-assume-safe-padding

# XHOST_FLAG = $(KEY)xCORE-AVX2 #-no-vec
# XHOST_FLAG = $(KEY)xSSE4.1 #-no-vec
# XHOST_FLAG = $(KEY)xMIC-AVX512 #-no-vec

CPLUS_FLAGS += $(QKEY)openmp-simd
CPLUS_FLAGS += $(FQKEY)MMD
ifneq ($(stdver),)
    CPLUS_FLAGS += $(FQKEY)std=$(stdver)
endif

ifeq ($(vecreport),$(filter $(vecreport),1 2 3 4 5 6 7))
    CPLUS_FLAGS +=  $(QKEY)opt-report=$(vecreport) $(QKEY)opt-report-phase vec $(QKEY)opt-report-phase:loop
else
    #suppress warning #15552: loop was not vectorized with "simd" and #15518: Structure assignment was serialized
    CPLUS_FLAGS += $(KEY)diag-disable:15552,15518
    ifeq (macos, $(os_name))
        CPLUS_FLAGS += $(QKEY)no-opt-report-embed #suppress "unaligned pointer" warnings
    endif
endif

OPTIMIZATION_DISABLED_FLAGS += $(KEY)debug inline-debug-info
