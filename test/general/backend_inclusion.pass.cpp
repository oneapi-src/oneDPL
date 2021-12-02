// -*- C++ -*-
//===-- backend_inclusion.pass.cpp ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

#if !_ONEDPL_PAR_BACKEND_SERIAL
#   ifdef _ONEDPL_PARALLEL_BACKEND_SERIAL_H
#        error The parallel serial backend is used while it should not (_ONEDPL_PAR_BACKEND_SERIAL==0)
#   endif
#endif

#if defined(ONEDPL_USE_OPENMP_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND
#   ifdef _ONEDPL_PARALLEL_BACKEND_OMP_H
#       error The parallel OpenMP backend is used while it should not (ONEDPL_USE_OPENMP_BACKEND==0)
#   endif
#endif

#if defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_TBB_BACKEND
#   ifdef _ONEDPL_PARALLEL_BACKEND_TBB_H
#       error The parallel TBB backend is used while it should not (ONEDPL_USE_TBB_BACKEND==0)
#   endif
#endif

#if defined(ONEDPL_USE_DPCPP_BACKEND) && !ONEDPL_USE_DPCPP_BACKEND
#   ifdef _ONEDPL_parallel_backend_sycl_H
#       error The parallel DPC++ backend is used while it should not (ONEDPL_USE_DPCPP_BACKEND==0)
#   endif
#endif

#if !_ONEDPL_FPGA_DEVICE
#   ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#       error The DPC++ backend for the FPGA is used while it should not (_ONEDPL_FPGA_DEVICE==0)
#   endif
#endif

int main() {
    return TestUtils::done();
}
