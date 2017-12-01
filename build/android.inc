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

PSTL_ARCH=
PIE_FLAGS = -pie -fPIE -fPIC
SDL_FLAGS = -fstack-protector -Wformat -Wformat-security
CPLUS_FLAGS += $(TARGET_CFLAGS) $(PIE_FLAGS) $(SDL_FLAGS)
# Paths to the NDK prebuilt tools and libraries
CPLUS_FLAGS += --sysroot=$(SYSROOT)
LDFLAGS += --sysroot=$(SYSROOT) $(TARGET_CFLAGS)
PSTL_LIB_LINK +=  -lc++abi -L$(CPLUS_LIB_PATH) -lc++_shared

ifeq (arm,$(arch))
    PSTL_LIB_LINK += -lunwind 
endif

# TARGET_CXX cames from NDK
override CPLUS:=$(TARGET_CXX) $(USE_SHARED_CPPRUNTIME_KEY)
LD = $(CPLUS) $(TARGET_CFLAGS) $(PIE_FLAGS) $(SDL_FLAGS)

run_cmd ?= -sh $(proj_root)/build/android.linux.launcher.sh $(largs)

# TBB_LIBRARIES := $(foreach dir,$(LIBRARY_PATH),$(wildcard $(dir)/libtbb*so))
TBB_LIBRARIES := $(foreach dir,$(LIBRARY_PATH),$(wildcard $(dir)/*))
LIB_STL_ANDROID += $(TBB_LIBRARIES)

$(warning LIB_STL_ANDROID=$(LIB_STL_ANDROID))
$(warning TBB_LIBRARIES=$(TBB_LIBRARIES))
$(warning LIBRARY_PATH=$(LIBRARY_PATH))
