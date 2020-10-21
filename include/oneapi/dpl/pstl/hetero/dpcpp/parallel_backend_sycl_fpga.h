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
#ifndef _ONEDPL_parallel_backend_sycl_fpga_H
#define _ONEDPL_parallel_backend_sycl_fpga_H

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

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

namespace sycl = cl::sycl;

namespace __ranges
{
//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------
//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.

template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_for(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs)
{
    auto __n = __get_first_range(::std::forward<_Ranges>(__rngs)...).size();
    assert(__n > 0);

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
#if __SYCL_UNNAMED_LAMBDA__
    using __kernel_name_t = __parallel_for_kernel<_Fp, __kernel_name, _Ranges...>;
#else
    using __kernel_name_t = __parallel_for_kernel<__kernel_name>;
#endif

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
        //get an access to data under SYCL buffer:
        oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

        __cgh.single_task<__kernel_name_t>([=]() mutable {
#pragma unroll(::std::decay <_ExecutionPolicy>::type::unroll_factor)
            for (auto __idx = 0; __idx < __count; ++__idx)
            {
                __brick(__idx, __rngs...);
            }
        });
    });
}

} //namespace __ranges

//------------------------------------------------------------------------
// parallel_transform_reduce
//------------------------------------------------------------------------

template <typename _Tp, typename _ExecutionPolicy, typename _Up, typename _Cp, typename _Rp, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Tp>
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _Up __u, _Cp __combine, _Rp __brick_reduce, _Ranges&&... __rngs)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_Tp>(
        __device_policy, __u, __combine, __brick_reduce, ::std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation, typename _InitType,
          typename _LocalScan, typename _GroupScan, typename _GlobalScan>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<
    _ExecutionPolicy, ::std::pair<oneapi::dpl::__internal::__difference_t<_Range2>, typename _InitType::__value_type>>
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _BinaryOperation __binary_op,
                          _InitType __init, _LocalScan __local_scan, _GroupScan __group_scan, _GlobalScan __global_scan)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        __device_policy, ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2), __binary_op, __init,
        __local_scan, __group_scan, __global_scan);
}

//------------------------------------------------------------------------
// __parallel_find_or
//-----------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Brick, typename _BrickTag, typename... _Ranges>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<
    _ExecutionPolicy,
    typename ::std::conditional<::std::is_same<_BrickTag, __parallel_or_tag>::value, bool,
                                oneapi::dpl::__internal::__difference_t<
                                    typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>::type>
__parallel_find_or(_ExecutionPolicy&& __exec, _Brick __f, _BrickTag __brick_tag, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(__device_policy, __f, __brick_tag,
                                                                 ::std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_or
//-----------------------------------------------------------------------
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
              _Iterator2 __s_last, _Brick __f)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_or(__device_policy, __first, __last, __s_first, __s_last, __f);
}

template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, bool>
__parallel_or(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_or(__device_policy, __first, __last, __f);
}

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _IsFirst>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator1>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first,
                _Iterator2 __s_last, _Brick __f, _IsFirst __is_first)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_find(__device_policy, __first, __last, __s_first, __s_last,
                                                              __f, __is_first);
}

template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator>
__parallel_find(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Brick __f, _IsFirst __is_first)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_find(__device_policy, __first, __last, __f, __is_first);
}

//------------------------------------------------------------------------
// parallel_merge
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Iterator3, typename _Compare>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, _Iterator3>
__parallel_merge(_ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1, _Iterator2 __first2,
                 _Iterator2 __last2, _Iterator3 __d_first, _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_merge(__device_policy, __first1, __last1, __first2, __last2,
                                                               __d_first, __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last, _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_stable_sort(__device_policy, __first, __last, __comp);
}

//------------------------------------------------------------------------
// parallel_partial_sort
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, void>
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_partial_sort(__device_policy, __first, __mid, __last, __comp);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_parallel_backend_sycl_H */
