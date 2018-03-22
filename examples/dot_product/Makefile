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

# GNU Makefile that builds and runs example.
run_cmd=
PROG=dot_product.exe
ARGS=

CXXFLAGS += -D__PSTL_USE_TBB -std=c++11

# Set by default icc as C++ compiler if it's present
ifneq (,$(shell which icc 2>/dev/null))
CXX = icc
endif # which icc

ifeq ($(CXX),icc)
CXXFLAGS += -qopenmp-simd
ifneq (, $(filter $(target), mic))
CXXFLAGS += -mmic
else
CXXFLAGS += -xHOST
endif # target is mic or host?
endif # icc?

ifeq ($(shell uname), Linux)
LIBS+= -lrt
else ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib
endif

all:    release test

release: *.cpp
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG=1 $(CXXFLAGS) -o $(PROG) $^ -ltbb_debug $(LIBS)

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)

perf_build: release

perf_run: test
