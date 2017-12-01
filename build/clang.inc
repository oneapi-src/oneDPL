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

override compiler:=clang++

ifneq ($(target),android)
    PSTL_ARCH += $(KEY)march=native
endif

XHOST_FLAG = -fno-vectorize
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
# XHOST_FLAG = $(KEY)mavx2 -fno-vectorize
# XHOST_FLAG = $(KEY)mavx512f -fno-vectorize
# DYN_LDFLAGS += $(LINK_KEY)c++ 
# CPLUS_FLAGS += -stdlib=libc++
# CPLUS_FLAGS += -fopenmp-simd //it will be supported in he future version
