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
#    include "parallel_backend_tbb.h"
#elif ONEDPL_USE_OPENMP_BACKEND || (!defined(ONEDPL_USE_OPENMP_BACKEND) && _ONEDPL_OPENMP_AVAILABLE)
#    define _ONEDPL_PAR_BACKEND_OPENMP 1
#    include "parallel_backend_omp.h"
#else
#    define _ONEDPL_PAR_BACKEND_SERIAL 1
#    include "parallel_backend_serial.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __backend
{

// Template for backend implementations
template <typename __backend_tag>
struct __backend_impl;

} // namespace __backend
} // namespace dpl
} // namespace oneapi

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/parallel_backend_sycl.h"
#    if _ONEDPL_FPGA_DEVICE
#        include "hetero/dpcpp/parallel_backend_sycl_fpga.h"
#    endif
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename __backend_tag>
using __par_backend = oneapi::dpl::__backend::__backend_impl<__backend_tag>;

template <typename __backend_tag, typename _ExecutionPolicy, typename _Tp>
using __par_backend_buffer = typename __par_backend<__backend_tag>::template __buffer<_ExecutionPolicy, _Tp>;

} // __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_H
