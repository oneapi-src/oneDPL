#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
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

SCAN_TARGET=$1

SKIP_PATTERN='*/.github/*'

# Ignored cases
IGNORE_COMMAND="sed -e /.*sycl_iterator.pass.cpp.*nd\\s*=.*/d \
-e /.*nanorange.hpp.*copyable\\s*=.*/d \
-e /.*iterator_impl.h.*Copyable\\s*=.*/d \
-e /.*windows.inc.*Od\\s*=.*/d \
-e /.*README.md.*varN\\s*=.*/d"

SCAN_RESULT=$(codespell --quiet-level=2 --skip "${SKIP_PATTERN}" ${SCAN_TARGET})
SCAN_RESULT=$(echo -e "${SCAN_RESULT}" | ${IGNORE_COMMAND})
echo "${SCAN_RESULT}"

if [[ ! -z ${SCAN_RESULT} ]]; then
    exit 1
fi
