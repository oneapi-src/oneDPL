##===-- CMakeLists.txt ----------------------------------------------------===##
#
# Copyright (C) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

# Required parameters:
#     VARS_TEMPLATE            - incoming path to PSTL offload vars template
#     OUTPUT_VARS              - path to vars file to be generated
#     PSTL_OFFLOAD_BINARY_PATH - path to PSTL offload binaries

set(DPLROOT_REPLACEMENT "${CMAKE_SOURCE_DIR}")
set(LIBRARY_PATH_REPLACEMENT "${PSTL_OFFLOAD_BINARY_PATH}")

configure_file(${VARS_TEMPLATE} ${OUTPUT_VARS} @ONLY)
