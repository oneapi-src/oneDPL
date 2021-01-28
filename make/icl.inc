##===-- icl.inc -----------------------------------------------------------===##
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

XHOST_FLAG = $(QKEY)xHOST #vec-
# XHOST_FLAG = $(QKEY)xCORE-AVX2
# XHOST_FLAG = $(QKEY)xSSE4.1
# XHOST_FLAG = $(QKEY)xMIC-AVX512

CPLUS_FLAGS += $(QKEY)opt-assume-safe-padding
CPLUS_FLAGS += $(QKEY)openmp-simd
CPLUS_FLAGS += $(FQKEY)MMD
ifneq ($(stdver),)
    CPLUS_FLAGS += $(FQKEY)std=$(stdver)
endif

ifeq ($(vecreport),$(filter $(vecreport),1 2 3 4 5 6 7))
    CPLUS_FLAGS +=  $(QKEY)opt-report=$(vecreport) $(QKEY)opt-report-phase vec $(QKEY)opt-report-phase:loop
else
    #suppress warning #15552: loop was not vectorized with "simd" and #15518: Structure assignment was serialized
    CPLUS_FLAGS += $(QKEY)diag-disable:15552,15518
endif

DISABLED_WARNINGS = $(QKEY)diag-disable:2586 #use comma-separated values to specify multiple entries
