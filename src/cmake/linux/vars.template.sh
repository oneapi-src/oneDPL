##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

export DPL_ROOT=@DPLROOT_REPLACEMENT@

LIBRARY_PATH="@LIBRARY_PATH_REPLACEMENT@:${LIBRARY_PATH}"; export LIBRARY_PATH
LD_LIBRARY_PATH="@LIBRARY_PATH_REPLACEMENT@:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
