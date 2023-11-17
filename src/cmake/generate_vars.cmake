##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

# Required parameters:
#     DPL_ROOT                  - value for DPLROOT environment variable
#     VARS_TEMPLATE            - incoming path to PSTL offload vars template
#     OUTPUT_VARS              - path to vars file to be generated
#     PSTL_OFFLOAD_BINARY_PATH - path to PSTL offload binaries

set(DPLROOT_REPLACEMENT "${DPL_ROOT}")
set(LIBRARY_PATH_REPLACEMENT "${PSTL_OFFLOAD_BINARY_PATH}")

configure_file(${VARS_TEMPLATE} ${OUTPUT_VARS} @ONLY)
