##===-- windows.inc -------------------------------------------------------===##
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

PYTHON = python
compiler ?= icl

KEY = /
FKEY = $(KEY)F
QKEY = $(KEY)Q
FQKEY = $(QKEY)
LD_OUT_KEY = $(KEY)OUT:
MACHINE_KEY = $(QKEY)
COMPILER_NOLOGO_KEY = $(KEY)nologo

OPTIMIZATION_DISABLED_FLAGS = $(KEY)Od $(KEY)Zi $(KEY)Fd"$*.pdb"

ifeq ($(cfg), debug)
ifeq (, $(filter $(compiler), dpcpp dpc++))
    LINK_KEY += $(KEY)debug
endif
    USE_SHARED_CPPRUNTIME_KEY += $(KEY)MDd $(KEY)EHsc
else
    USE_SHARED_CPPRUNTIME_KEY += $(KEY)MD $(KEY)EHsc
endif

LD = LINK
OBJ_SFX = .obj
DEL_CMD = del $(KEY)F
RUN_CMD =
COMMAND_SEPARATOR = &
OPTIMIZATION_DISABLED_FLAGS += $(KEY)DEBUG
OPTIMIZATION_ENABLED_FLAGS = $(KEY)O2 $(KEY)DNDEBUG

ifneq ($(PSTL_LIB_NAME), )
    PSTL_LIB_EXT = .lib
    PSTL_LIB_LINK += $(LINK_KEY)$(PSTL_LIB_NAME)$(PSTL_LIB_EXT)
endif
# Do not update LDFLAGS with corresponding TBB_LIB_NAME here, because of
# implicit linkage capability of TBB library

ifneq (, $(filter $(compiler), cl icl dpcpp dpc++))
    CPLUS_FLAGS += $(KEY)bigobj
endif

ifeq ($(compiler), cl)
    ifneq ($(stdver),)
        CPLUS_FLAGS += $(KEY)std:$(stdver)
    endif
endif

DYN_LDFLAGS += $(LINK_KEY)

ifneq (,$(filter uwp,$(target_app) $(target_ui)))
    CPLUS_FLAGS += $(KEY)ZW:nostdlib $(KEY)D "_UNICODE" $(KEY)D "UNICODE" $(KEY)D "WINAPI_FAMILY=WINAPI_FAMILY_APP"
    _WIN32_WINNT = 0x0A00
    DYN_LDFLAGS += $(KEY)NODEFAULTLIB:"kernel32.lib" OneCore.lib
endif
