##===-- dpcpp.inc ---------------------------------------------------------===##
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

ifeq ($(os_name), linux)
    DPCPP_OPTIMIZATION_DISABLED_FLAGS=$(KEY)O1 $(KEY)g
    ifneq (, $(shell which dpcpp))
        override compiler:=dpcpp
    endif
    CPLUS_FLAGS += -fopenmp-simd
    ifeq ($(compiler), clang++)
        CPLUS_FLAGS += -fsycl
        DYN_LDFLAGS += -fsycl
    endif
    ifeq ($(use_unnamed_lambda), 1)
        CPLUS_FLAGS += -fsycl-unnamed-lambda
    endif
    ifneq ($(stdver),)
        CPLUS_FLAGS += $(KEY)std=$(stdver)
    endif
    ifeq ($(backend), sycl)
        DYN_LDFLAGS += -ltbb
    endif
endif

ifeq ($(os_name), windows)
    DPCPP_OPTIMIZATION_DISABLED_FLAGS=$(KEY)O1 $(KEY)Zi $(KEY)Fd"$*.pdb" $(KEY)DEBUG
    override compiler:=dpcpp
    CPLUS_FLAGS += -qopenmp-simd
    ifneq ($(stdver),)
        CPLUS_FLAGS += $(QKEY)std:$(stdver)
    endif
    ifeq ($(backend), sycl)
        DYN_LDFLAGS += /Qtbb
    endif
endif

# In case of TBB/SYCL backend and using Clang with libstdc++ the version of the library should be specify explicitly via TBB_USE_GLIBCXX_VERSION
ifneq (, $(filter $(backend), sycl tbb))
    export gcc_version:=$(shell gcc -dumpfullversion -dumpversion)
    ifneq (,$(gcc_version))
        # TODO: do not assume that GCC minor and patchlevel versions are always single-digit.
        CPLUS_FLAGS += $(KEY)DTBB_USE_GLIBCXX_VERSION=$(subst .,0,$(gcc_version))
    else
        CPLUS_FLAGS += $(KEY)DTBB_USE_GLIBCXX_VERSION=70300
    endif
endif

ifneq ($(findstring FPGA, $(device_type)),)
    DEVICE_COMPILE_FLAGS += -fintelfpga
    DEVICE_COMPILE_FLAGS += $(KEY)DONEDPL_FPGA_DEVICE
endif

ifeq ($(device_type), FPGA)
    DEVICE_LINK_FLAGS += -Xshardware
    ifneq ($(static_report), 0)
        DEVICE_LINK_FLAGS += -fsycl-link
    endif
endif

ifeq ($(device_type), FPGA_EMULATOR)
    DEVICE_COMPILE_FLAGS += $(KEY)DONEDPL_FPGA_EMULATOR
endif
ifeq ($(ranges_api), 1)
    CPLUS_FLAGS += $(KEY)D_ENABLE_RANGES_TESTING=1
endif
