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

vecreport ?= 0 #may be set to [0..7], see https://software.intel.com/en-us/node/522949 for details

XHOST_FLAG = $(KEY)xHOST -no-vec
CPLUS_FLAGS += $(QKEY)opt-assume-safe-padding

# XHOST_FLAG = $(KEY)xCORE-AVX2 -no-vec
# XHOST_FLAG = $(KEY)xSSE4.1 -no-vec
# XHOST_FLAG = $(KEY)xMIC-AVX512 -no-vec

CPLUS_FLAGS += $(QKEY)openmp-simd
CPLUS_FLAGS += $(FQKEY)MMD
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
CPLUS_FLAGS +=  $(QKEY)opt-report=$(vecreport) $(QKEY)opt-report-phase vec

OPTIMIZATION_DISABLED_FLAGS += $(KEY)debug inline-debug-info
