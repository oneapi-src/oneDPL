// -*- C++ -*-
//===----------------------------------------------------------------------===//
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
#ifndef _ONEDPL_PARALLEL_BACKEND_H
#define _ONEDPL_PARALLEL_BACKEND_H
#include "onedpl_config.h"

// Select a parallel backend
#if ONEDPL_USE_TBB_BACKEND || (!defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND && _ONEDPL_TBB_AVAILABLE)
#    define _ONEDPL_PAR_BACKEND_TBB 1
#endif

#if ONEDPL_USE_OPENMP_BACKEND || (!defined(ONEDPL_USE_OPENMP_BACKEND) && _ONEDPL_OPENMP_AVAILABLE)
#    define _ONEDPL_PAR_BACKEND_OPENMP 1
#endif

#if !_ONEDPL_PAR_BACKEND_TBB && !_ONEDPL_PAR_BACKEND_OPENMP
#    define _ONEDPL_PAR_BACKEND_SERIAL 1
#endif

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/parallel_backend_sycl.h"
#    if _ONEDPL_FPGA_DEVICE
#        include "hetero/dpcpp/parallel_backend_sycl_fpga.h"
#    endif
#endif

#if _ONEDPL_PAR_BACKEND_TBB
#    include "parallel_backend_tbb.h"
namespace oneapi
{
namespace dpl
{
namespace __par_backend = __tbb_backend;
}
} // namespace oneapi
#elif _ONEDPL_PAR_BACKEND_OPENMP
#    include "parallel_backend_omp.h"
namespace oneapi
{
namespace dpl
{
namespace __par_backend = __omp_backend;
}
} // namespace oneapi
#elif _ONEDPL_PAR_BACKEND_SERIAL
#    include "parallel_backend_serial.h"
namespace oneapi
{
namespace dpl
{
namespace __par_backend = __serial_backend;
}
} // namespace oneapi
#else
_PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif // _ONEDPL_PARALLEL_BACKEND_H
