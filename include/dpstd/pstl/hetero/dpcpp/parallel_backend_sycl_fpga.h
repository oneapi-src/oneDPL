// -*- C++ -*-
//===-- parallel_backend_sycl_fpga.h --------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

//!!! NOTE: This file should be included under the macro _PSTL_BACKEND_SYCL
#ifndef _PSTL_parallel_backend_sycl_fpga_H
#define _PSTL_parallel_backend_sycl_fpga_H

#include <CL/sycl.hpp>

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <iostream>

#include "parallel_backend_sycl_utils.h"
// workaround until we implement more performant optimization for patterns
#include "parallel_backend_sycl.h"
#include "../../execution_impl.h"
#include "execution_sycl_defs.h"
#include "../../iterator_impl.h"
#include "sycl_iterator.h"

namespace dpstd
{
namespace __par_backend_hetero
{

namespace sycl = cl::sycl;

//------------------------------------------------------------------------
// parallel_for_ext
//------------------------------------------------------------------------
//Extended version of parallel_for, one additional parameter was added to control
//size of __target__buffer. For some algorithms happens that size of
//processing range is n, but amount of iterations is n/2.

template <typename _ExecutionPolicy, typename _Iterator, typename _Fp>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_for_ext(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last, _Fp __brick)
{
    if (__first == __last || __first == __mid)
        return;
    auto __target_buffer = __internal::get_buffer()(__first, __last); // hides sycl::buffer
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_for_kernel<_Iterator, _Fp, __kernel_name>;
#else
    using __kernel_name_t = __parallel_for_kernel<__kernel_name>;
#endif

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    //In reverse we need (__mid - __first) swaps
    auto __n = __mid - __first;
    __exec.queue().submit([&__target_buffer, &__brick, __n](sycl::handler& __cgh) {
        auto __acc = __internal::get_access<_Iterator>(__cgh)(__target_buffer);

        __cgh.single_task<__kernel_name_t>([=]() mutable {
#pragma unroll(std::decay <_ExecutionPolicy>::type::unroll_factor)
            for (auto __idx = 0; __idx < __n; ++__idx)
            {
                __brick(__idx, __acc);
            }
        });
    });
}

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Fp>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_for(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Fp __brick)
{
    __parallel_for_ext(std::forward<_ExecutionPolicy>(__exec), __first, __last, __last, __brick);
}

//------------------------------------------------------------------------
// parallel_transform_reduce
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Iterator, typename _Up, typename _Cp, typename _Rp,
          typename... _PolicyParams>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Tp>
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Up __u, _Cp __combine,
                            _Rp __brick_reduce)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_transform_reduce<_Tp>(__device_policy, __first, __last, __u, __combine, __brick_reduce);
}

//------------------------------------------------------------------------
// parallel_transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _BinaryOperation, typename _Tp,
          typename _Transform, typename _Reduce, typename _Scan>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, std::pair<_Iterator2, _Tp>>
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                          _BinaryOperation __binary_op, _Tp __init, _Transform __brick_transform,
                          _Reduce __brick_reduce, _Scan __brick_scan)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_transform_scan(__device_policy, __first, __last, __result, __binary_op, __init, __brick_transform,
                                     __brick_reduce, __brick_scan);
}

//------------------------------------------------------------------------
// parallel_or
//-----------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
              _Iterator2 __s_last, _Brick __f)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_or(__device_policy, __first, __last, __s_first, __s_last, __f);
}

template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f)
{
    return dpstd::__par_backend_hetero::__parallel_or(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                      /* dummy */ __first, __first, __f);
}

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _IsFirst>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator1>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                _Iterator2 __s_last, _Brick __f, _IsFirst __is_first)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_find(__device_policy, __first, __last, __s_first, __s_last, __f, __is_first);
}

template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f, _IsFirst __is_first)
{
    return dpstd::__par_backend_hetero::__parallel_find(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                        /*dummy*/ __first, __first, __f, __is_first);
}

//------------------------------------------------------------------------
// parallel_merge
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Iterator3, typename _Compare>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator3>
__parallel_merge(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                 _Iterator2 __last2, _Iterator3 __d_first, _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_merge(__device_policy, __first1, __last1, __first2, __last2, __d_first, __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_stable_sort(__device_policy, __first, __last, __comp);
}

//------------------------------------------------------------------------
// parallel_partial_sort
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
dpstd::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = dpstd::execution::make_device_policy<__kernel_name>(__exec.queue());
    return __parallel_partial_sort(__device_policy, __first, __mid, __last, __comp);
}

} // namespace __par_backend_hetero
} // namespace dpstd

#endif /* _PSTL_parallel_backend_sycl_H */
