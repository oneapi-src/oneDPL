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

// Verify that deactivated backends are not accessed, by checking respective header guard macros.
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

// Verify that the guard must be undefined if ONEDPL_FPGA_DEVICE is undefined or set to zero.
#if !ONEDPL_FPGA_DEVICE
#   ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#       error The DPC++ backend for the FPGA is used while it should not (ONEDPL_FPGA_DEVICE==0)
#   endif
#endif

// Verify that the backend is selected according to the table below and that the backends are mutually exclusive.
//  ___________________________________________________________
// |       \      |                     |           |          |
// | OpenMP \ TBB |          0          | Undefined |    1     |
// |_________\____|_____________________|___________|__________|
// |              |                     |           |          |
// |      0       |       Serial        |    TBB    |   TBB    |
// |______________|_____________________|___________|__________|
// |              | OpenMP if available |           |          |
// |  Undefined   |  Otherwise, serial  |    TBB    |   TBB    |
// |______________|____________________ |___________|__________|
// |              |                     |           |          |
// |       1      |       OpenMP        |  OpenMP   |   TBB    |
// |______________|_____________________|___________|__________|

#if defined(ONEDPL_USE_TBB_BACKEND) && ONEDPL_USE_TBB_BACKEND
#   if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#       error The parallel TBB backend is not used while it should (ONEDPL_USE_TBB_BACKEND==1 and TBB backend in priority)
#   endif
#   if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#       error The parallel OpenMP backend is used while it should not (ONEDPL_USE_TBB_BACKEND==1 and TBB backend in priority)
#   endif
#   if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#       error The parallel serial backend is used while it should not (ONEDPL_USE_TBB_BACKEND==1 and TBB in priority)
#   endif
#endif

#if defined(ONEDPL_USE_OPENMP_BACKEND) && ONEDPL_USE_OPENMP_BACKEND && !ONEDPL_USE_TBB_BACKEND
#   if defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#       error The parallel TBB backend is used while it should not (ONEDPL_USE_OPENMP_BACKEND==1 and OpenMP backend in priority)
#   endif
#   if !defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#       error The parallel OpenMP backend is not used while it should (ONEDPL_USE_OPENMP_BACKEND==1 and OpenMP backend in priority)
#   endif
#   if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#       error The parallel serial backend is used while it should not (ONEDPL_USE_OPENMP_BACKEND==1 and OpenMP backend in priority)
#   endif
#endif

#if defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_TBB_BACKEND
#   if defined(ONEDPL_USE_OPENMP_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND
#       if !defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#           error The parallel serial backend is not used while it should (Serial backend in priority)
#       endif
#   elif defined(_OPENMP)
#       if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#           error The parallel TBB backend is not used while it should (TBB backend in priority)
#       endif
#       if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#           error The parallel OpenMP backend is used while it should not (TBB backend in priority)
#       endif
#       if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#           error The parallel serial backend is used while it should not (TBB backend in priority)
#       endif
#    else
#       if defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#           error The parallel TBB backend is used while it should not (Serial backend in priority)
#       endif
#       if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#           error The parallel OpenMP backend is used while it should not (Serial backend in priority)
#       endif
#       if !defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#           error The parallel serial backend is not used while it should (Serial backend in priority)
#       endif
#   endif
#else
#   if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#       error The parallel TBB backend is not used while it should (TBB backend in priority)
#   endif
#   if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#       error The parallel OpenMP backend is used while it should not (TBB backend in priority)
#   endif
#   if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#       error The parallel serial backend is used while it should not (TBB backend in priority)
#   endif
#endif

int main() {
    return TestUtils::done();
}
