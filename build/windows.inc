# Copyright (c) 2017 Intel Corporation
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
compiler ?= icl
LD = LINK
KEY = /
LINK_KEY =
QKEY = $(KEY)Q
FKEY = $(KEY)F
FQKEY = $(QKEY)
MACHINE_KEY = $(QKEY)
OBJ_SFX = .obj
DEL_CMD = del $(KEY)F
RUN_CMD =
COMMAND_SEPARATOR = &
COMPILER_NOLOGO_KEY = $(KEY)nologo
OPTIMIZATION_DISABLED_FLAGS = $(KEY)Od $(KEY)Zi $(KEY)DEBUG $(KEY)Fd"$*.pdb"
OPTIMIZATION_ENABLED_FLAGS = $(KEY)O2 $(KEY)DNDEBUG
LD_OUT_KEY = $(KEY)OUT:

ifneq ($(PSTL_LIB_NAME), )
    PSTL_LIB_EXT = .lib
    PSTL_LIB_LINK += $(LINK_KEY)$(PSTL_LIB_NAME)$(PSTL_LIB_EXT)
endif
# Do not update LDFLAGS with corresponding TBB_LIB_NAME here, because of
# implicit linkage capability of TBB library

ifeq ($(cfg),debug)
    LINK_KEY += $(KEY)debug 
    USE_SHARED_CPPRUNTIME_KEY += $(KEY)MDd $(KEY)EHsc 
    BACKEND_MACRO += -DTBB_USE_DEBUG=1
else
    USE_SHARED_CPPRUNTIME_KEY += $(KEY)MD $(KEY)EHsc 
endif

ifneq (, $(filter $(compiler), cl icl))
    CPLUS_FLAGS += $(KEY)bigobj
endif


DYN_LDFLAGS += $(LINK_KEY)

ifneq (,$(filter uwp,$(target_app) $(target_ui)))
    CPLUS_FLAGS += /ZW:nostdlib /D "_UNICODE" /D "UNICODE" /D "WINAPI_FAMILY=WINAPI_FAMILY_APP"
    _WIN32_WINNT = 0x0A00
    DYN_LDFLAGS += /NODEFAULTLIB:"kernel32.lib" OneCore.lib
endif
