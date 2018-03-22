#!/bin/sh
#
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

# Parsing script arguments
# Arg1 represents target architecture. Its possible values are 'ia32' or 'intel64',
# default value equals to the value of $COMPILERVARS_ARCHITECTURE environment variable.

PSTL_TARGET_ARCH=

if [ -n "${COMPILERVARS_ARCHITECTURE}" ]; then
    PSTL_TARGET_ARCH=$COMPILERVARS_ARCHITECTURE
fi

if [ -n "$1" ]; then
    PSTL_TARGET_ARCH=$1
fi

if [ -n "${PSTL_TARGET_ARCH}" ]; then
    if [ "$PSTL_TARGET_ARCH" != "ia32" -a "$PSTL_TARGET_ARCH" != "intel64" ]; then
        echo "ERROR: Unknown switch '$PSTL_TARGET_ARCH'. Accepted values: ia32, intel64"
        PSTL_TARGET_ARCH=
        return 1;
    fi
else
    echo "ERROR: Architecture is not defined. Accepted values: ia32, intel64"
    return 1;
fi

# Arg2 represents PSTLROOT detection method. Its possible value is 'auto_pstlroot'. In which case
# the environment variable PSTLROOT is detected automatically by using the script directory path.
PSTLROOT=SUBSTITUTE_INSTALL_DIR_HERE
if [ -n "${BASH_SOURCE}" ]; then
    if [ "$2" = "auto_pstlroot" ]; then
       PSTLROOT=$(cd $(dirname ${BASH_SOURCE}) && pwd -P)/..
    fi
fi
export PSTLROOT

if [ -e $PSTLROOT/../tbb/bin/tbbvars.sh ]; then
   . $PSTLROOT/../tbb/bin/tbbvars.sh $PSTL_TARGET_ARCH
fi

if [ -z "${CPATH}" ]; then
    CPATH="${PSTLROOT}/include"; export CPATH
else
    CPATH="${PSTLROOT}/include:$CPATH"; export CPATH
fi
