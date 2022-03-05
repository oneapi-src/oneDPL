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
#    ifdef _ONEDPL_PARALLEL_BACKEND_SERIAL_H
#        error The serial backend is used while it should not (_ONEDPL_PAR_BACKEND_SERIAL == 0)
#    endif
#endif

#if defined(ONEDPL_USE_OPENMP_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND
#    ifdef _ONEDPL_PARALLEL_BACKEND_OMP_H
#        error The OpenMP backend is used while it should not (ONEDPL_USE_OPENMP_BACKEND == 0)
#    endif
#endif

#if defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_TBB_BACKEND
#    ifdef _ONEDPL_PARALLEL_BACKEND_TBB_H
#        error The TBB backend is used while it should not (ONEDPL_USE_TBB_BACKEND == 0)
#    endif
#endif

#if defined(ONEDPL_USE_DPCPP_BACKEND) && !ONEDPL_USE_DPCPP_BACKEND
#    ifdef _ONEDPL_parallel_backend_sycl_H
#        error The DPC++ backend is used while it should not (ONEDPL_USE_DPCPP_BACKEND == 0)
#    endif
#    ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#        error The DPC++ backend for the FPGA is used while it should not (ONEDPL_USE_DPCPP_BACKEND == 0)
#    endif
#endif

// Verify that DPC++ backend for the FPGA is not not accessed if ONEDPL_FPGA_DEVICE is undefined or set to 0.
#if !ONEDPL_FPGA_DEVICE
#    ifdef _ONEDPL_parallel_backend_sycl_fpga_H
#        error The DPC++ backend for the FPGA is used while it should not (ONEDPL_FPGA_DEVICE==0)
#    endif
#endif

// Verify there is only one backend selected and the selection in the table below.
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

// Make sure that the TBB backend is selected if ONEDPL_USE_TBB_BACKEND set to 1 and ONEDPL_USE_OPENMP_BACKEND set to any value
#if defined(ONEDPL_USE_TBB_BACKEND) && ONEDPL_USE_TBB_BACKEND
#    if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#        error The TBB backend is not enabled while it should (ONEDPL_USE_TBB_BACKEND == 1)
#    endif
#    if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#        error The OpenMP backend cannot be simultaneously enabled with the TBB backend (ONEDPL_USE_TBB_BACKEND == 1)
#    endif
#    if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#        error The serial backend cannot be simultaneously enabled with the TBB backend (ONEDPL_USE_TBB_BACKEND == 1)
#    endif
#endif

// Make sure that the OpenMP backend is selected if ONEDPL_USE_OPENMP_BACKEND is set to 1
// and ONEDPL_USE_TBB_BACKEND is undefined or set to 0
#if defined(ONEDPL_USE_OPENMP_BACKEND) && ONEDPL_USE_OPENMP_BACKEND && !ONEDPL_USE_TBB_BACKEND
#    if defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#        error The TBB backend cannot be simultaneously enabled with the OpenMP backend (ONEDPL_USE_OPENMP_BACKEND == 1)
#    endif
#    if !defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#        error The OpenMP backend is not enabled while it should (ONEDPL_USE_OPENMP_BACKEND == 1)
#    endif
#    if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#        error The serial backend cannot be simultaneously enabled with the OpenMP backend (ONEDPL_USE_OPENMP_BACKEND == 1)
#    endif
#endif

#if defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_TBB_BACKEND
// Make sure that the serial backend is selected if ONEDPL_USE_OPENMP_BACKEND and ONEDPL_USE_TBB_BACKEND are set to 0
#    if defined(ONEDPL_USE_OPENMP_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND
#        if !defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#            error The serial backend is not enabled while it should because all parallel backends are disabled
#        endif
#    elif !defined(ONEDPL_USE_OPENMP_BACKEND)
// Make sure that the OpenMP backend is selected if ONEDPL_USE_OPENMP_BACKEND is undefined,
// ONEDPL_USE_TBB_BACKEND is set to 0 and OpenMP is available
#        if defined(_OPENMP)
#            if defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#                error The TBB backend cannot be simultaneously enabled with the OpenMP backend
#            endif
#            if !defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#                error The OpenMP backend is not enabled while it should (ONEDPL_USE_TBB_BACKEND == 0 && defined(_OPENMP))
#            endif
#            if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#                error The serial backend cannot be simultaneously enabled with the OpenMP backend
#            endif
// Make sure that the serial backend is selected if ONEDPL_USE_OPENMP_BACKEND is undefined,
// ONEDPL_USE_TBB_BACKEND is set to 0 and OpenMP is not available
#        else
#            if defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#                error The TBB backend cannot be simultaneously enabled with the serial backend
#            endif
#            if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#                error The OpenMP backend cannot be simultaneously enabled with the serial backend
#            endif
#            if !defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#                error The serial backend is not enabled while it should (ONEDPL_USE_TBB_BACKEND == 0 && !defined(_OPENMP))
#            endif
#        endif
#    endif
#endif

// Make sure that the TBB backend is selected if ONEDPL_USE_OPENMP_BACKEND is undefined or set to 0
// and ONEDPL_USE_TBB_BACKEND is undefined
#if !defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND
#    if !defined(_ONEDPL_PARALLEL_BACKEND_TBB_H)
#        error The TBB backend is not enabled while it should when neither of parallel backends is explicitly specified
#    endif
#    if defined(_ONEDPL_PARALLEL_BACKEND_OMP_H)
#        error The OpenMP backend cannot be simultaneously enabled with the TBB backend
#    endif
#    if defined(_ONEDPL_PARALLEL_BACKEND_SERIAL_H)
#        error The serial backend cannot be simultaneously enabled with the TBB backend
#    endif
#endif

int main() {
    return TestUtils::done();
}
