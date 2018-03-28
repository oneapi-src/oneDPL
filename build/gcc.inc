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

override compiler:=g++
XHOST_FLAG = $(KEY)march=native -fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx2 -fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx512f -fno-tree-vectorize
 DYN_LDFLAGS += $(LINK_KEY)stdc++
# GCC 4.8.5 and early doesn't  support -fopenmp-simd; GCC 4.9 supports OpenMP 4.0 for C/C++
ifneq (, $(shell gcc -dumpversion | egrep  "^4\.9\.[0-9]"))
    CPLUS_FLAGS += -fopenmp-simd
endif
ifneq (, $(shell gcc -dumpversion | egrep  "^[5-9]\.[0-9]\.[0-9]"))
    CPLUS_FLAGS += -fopenmp-simd
#   CPLUS_FLAGS += -fdump-rtl-loop2 #use this option to enable optimization report
endif

CPLUS_FLAGS += $(FQKEY)std=$(stdver)
