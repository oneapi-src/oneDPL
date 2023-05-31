// -*- C++ -*-
//===-- parallel_backend_sycl_fpga.h --------------------------------------===//
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

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL

// This header guard is used to check inclusion of DPC++ backend for the FPGA.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H

#include <cassert>
#include <algorithm>
#include <type_traits>

#include "sycl_defs.h"
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
//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------
//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.

// Please see the comment for __parallel_for_submitter for optional kernel name explanation
template <typename _Name>
struct __parallel_for_fpga_submitter;

template <typename... _Name>
struct __parallel_for_fpga_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs) const
    {
        auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
        assert(__n > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        auto __event = __exec.queue().submit([&__rngs..., &__brick, __count](sycl::handler& __cgh) {
            //get an access to data under SYCL buffer:
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

            __cgh.single_task<_Name...>([=]() {
#pragma unroll(::std::decay <_ExecutionPolicy>::type::unroll_factor)
                for (auto __idx = 0; __idx < __count; ++__idx)
                {
                    __brick(__idx, __rngs...);
                }
            });
        });
        return __future(__event);
    }
};

template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_for(_ExecutionPolicy&& __exec, _Fp __brick, _Index __count, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __parallel_for_name = __internal::__kernel_name_provider<typename _Policy::kernel_name>;

    return __parallel_for_fpga_submitter<__parallel_for_name>()(std::forward<_ExecutionPolicy>(__exec), __brick,
                                                                __count, std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_transform_reduce
//------------------------------------------------------------------------

template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _ExecutionPolicy, typename _InitType,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                            _InitType __init, _Ranges&&... __rngs)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_Tp>(
        __device_policy, __reduce_op, __transform_op, __init, ::std::forward<_Ranges>(__rngs)...);
}

//------------------------------------------------------------------------
// parallel_transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __in_rng, _Range2&& __out_rng, ::std::size_t __n,
                          _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        ::std::move(__device_policy), ::std::forward<_Range1>(__in_rng), ::std::forward<_Range2>(__out_rng), __n,
        __unary_op, __init, __binary_op, _Inclusive{});
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation, typename _InitType,
          typename _LocalScan, typename _GroupScan, typename _GlobalScan,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_transform_scan_base(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                               _BinaryOperation __binary_op, _InitType __init, _LocalScan __local_scan,
                               _GroupScan __group_scan, _GlobalScan __global_scan)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_transform_scan_base(
        std::move(__device_policy), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2), __binary_op,
        __init, __local_scan, __group_scan, __global_scan);
}

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _Pred,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_copy_if(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n, _Pred __pred)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_copy_if(
        ::std::move(__device_policy), ::std::forward<_InRng>(__in_rng), ::std::forward<_OutRng>(__out_rng), __n,
        __pred);
}


template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _CreateMaskOp,
          typename _CopyByMaskOp,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_scan_copy(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n,
                     _CreateMaskOp __create_mask_op, _CopyByMaskOp __copy_by_mask_op)
{
    // workaround until we implement more performant version for patterns
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    auto __device_policy = oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
    return oneapi::dpl::__par_backend_hetero::__parallel_scan_copy(
        ::std::move(__device_policy), ::std::forward<_InRng>(__in_rng), ::std::forward<_OutRng>(__out_rng), __n,
        __create_mask_op, __copy_by_mask_op);
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

template <typename _ExecutionPolicy>
auto
__device_policy(_ExecutionPolicy&& __exec)
    -> decltype(oneapi::dpl::execution::make_device_policy<typename ::std::decay<_ExecutionPolicy>::type::kernel_name>(
        __exec.queue()))
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using __kernel_name = typename _Policy::kernel_name;
    return oneapi::dpl::execution::make_device_policy<__kernel_name>(__exec.queue());
}

//------------------------------------------------------------------------
// parallel_merge
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare>
auto
__parallel_merge(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __rng3, _Compare __comp)
    -> oneapi::dpl::__internal::__enable_if_fpga_execution_policy<
        _ExecutionPolicy, decltype(oneapi::dpl::__par_backend_hetero::__parallel_merge(
                              __device_policy(__exec), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
                              ::std::forward<_Range3>(__rng3), __comp))>
{
    // workaround until we implement more performant version for patterns
    return oneapi::dpl::__par_backend_hetero::__parallel_merge(__device_policy(__exec), ::std::forward<_Range1>(__rng1),
                                                               ::std::forward<_Range2>(__rng2),
                                                               ::std::forward<_Range3>(__rng3), __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort
//-----------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_stable_sort(_ExecutionPolicy&& __exec, _Range&& __rng, _Compare __comp, _Proj __proj)
{
    // workaround until we implement more performant version for patterns
    return oneapi::dpl::__par_backend_hetero::__parallel_stable_sort(__device_policy(__exec),
                                                                     ::std::forward<_Range>(__rng), __comp, __proj);
}

//------------------------------------------------------------------------
// parallel_partial_sort
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_ExecutionPolicy, int> = 0>
auto
__parallel_partial_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid, _Iterator __last,
                        _Compare __comp)
{
    // workaround until we implement more performant version for patterns
    return oneapi::dpl::__par_backend_hetero::__parallel_partial_sort(__device_policy(__exec), __first, __mid, __last,
                                                                      __comp);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_FPGA_H
