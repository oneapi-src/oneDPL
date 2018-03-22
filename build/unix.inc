# Copyright (c) 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

PYTHON = python
KEY = -
QKEY = $(KEY)q
FKEY = $(KEY)
FQKEY = $(KEY)
MACHINE_KEY = $(KEY)m
OBJ_SFX = .o
DEL_CMD = rm $(KEY)f
RUN_CMD = ./
COMMAND_SEPARATOR = ;
compiler ?= icc
COMPILER_NOLOGO_KEY =
OPTIMIZATION_DISABLED_FLAGS = $(KEY)O0 $(KEY)g
OPTIMIZATION_ENABLED_FLAGS +=  $(KEY)O2
TBB_LIB_NAME = tbb
CPLUS = $(compiler)
LD = $(CPLUS)

USE_SHARED_CPPRUNTIME_KEY =
LINK_KEY = $(KEY)l

LD_OUT_KEY = $(KEY)o
DYN_LDFLAGS += -L. -L$(proj_root)/build
use_gold_linker ?= 0 # Use "gold" linker for test_algorithm when it runs on FreeBSD* 11 to avoid compilation errors

ifeq ($(use_gold_linker), 1)
    ifeq ($(cfg), debug)
        test_algorithm.exe: DYN_LDFLAGS += $(FQKEY)fuse-ld=gold
    endif
endif

ifneq ($(PSTL_LIB_NAME), )
    PSTL_LIB_LINK += $(LINK_KEY)$(PSTL_LIB_NAME)$(PSTL_LIB_EXT)
endif

ifneq (, $(filter $(backend), tbb))
    DYN_LDFLAGS += $(LINK_KEY)$(TBB_LIB_NAME)
endif


ifeq ($(arch),intel64)
    PSTL_ARCH = $(MACHINE_KEY)64
else ifeq ($(arch),ia32)
    PSTL_ARCH = $(MACHINE_KEY)32
else ifeq ($(arch),)
    $(info arch=native by default)
else
    PSTL_ARCH = $(MACHINE_KEY)$(arch)
endif
