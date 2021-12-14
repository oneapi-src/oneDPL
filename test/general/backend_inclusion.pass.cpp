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

// The following test suites verify that the unused backend is disabled.
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
#   ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#       error The DPC++ backend for the FPGA is used while it should not (ONEDPL_USE_DPCPP_BACKEND==0)
#   endif
#endif

#if !ONEDPL_FPGA_DEVICE
#   ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#       error The DPC++ backend for the FPGA is used while it should not (ONEDPL_FPGA_DEVICE==0)
#   endif
#endif

// The following test suites verify that the required backend is enabled.
#if defined(ONEDPL_USE_TBB_BACKEND) && ONEDPL_USE_TBB_BACKEND
#   if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#       error The parallel TBB backend is not used while it should (ONEDPL_USE_TBB_BACKEND==1)
#   endif
#elif defined(ONEDPL_USE_OPENMP_BACKEND) && ONEDPL_USE_OPENMP_BACKEND
#   if !defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#       error The parallel OpenMP backend is not used while it should (ONEDPL_USE_OPENMP_BACKEND==1)
#   endif
#endif

int main() {
    return TestUtils::done();
}
