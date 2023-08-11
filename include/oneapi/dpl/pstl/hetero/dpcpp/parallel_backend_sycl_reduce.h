// -*- C++ -*-
//===-- parallel_backend_sycl_reduce.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// Each work item transforms and reduces __iters_per_work_item elements from global memory and stores the result in SLM.
// 32 __iters_per_work_item was empirically found best for typical devices.
// Each work group of size __work_group_size reduces the preliminary results of each work item in a group reduction
// using SLM. 256 __work_group_size was empirically found best for typical devices.
constexpr ::std::uint16_t __reduce_work_group_size = 256;
constexpr ::std::uint8_t __reduce_iters_per_work_item = 32;
// The single work group implementation can process up to __reduce_work_group_size * __reduce_iters_per_work_item
// elements.
constexpr ::std::uint16_t __reduce_max_n_small = __reduce_work_group_size * __reduce_iters_per_work_item;
// The two-step tree reduction can process __reduce_work_group_size * __reduce_iters_per_work_item elements per step.
constexpr ::std::uint32_t __reduce_max_n_mid = __reduce_max_n_small * __reduce_max_n_small;

template <typename... _Name>
class __reduce_small_kernel;

template <typename... _Name>
class __reduce_mid_device_kernel;

template <typename... _Name>
class __reduce_mid_work_group_kernel;

template <typename... _Name>
class __reduce_kernel;

// Single work group kernel that transforms and reduces __n elements to the single result.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _InitType, typename _AccLocal, typename _Res, typename... _Acc>
void
__work_group_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __n_items,
                           _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern, _InitType __init,
                           const _AccLocal& __local_mem, const _Res& __res_acc, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, /*global_offset*/ (_Size)0, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    // 2. Reduce within work group using local memory
    _Tp __result = __reduce_pattern(__item_id, __n_items, __local_mem);
    if (__local_idx == 0)
    {
        __reduce_pattern.apply_init(__init, __result);
        __res_acc[0] = __result;
    }
}

// Device kernel that transforms and reduces __n elements to the number of work groups preliminary results.
template <typename _Tp, typename _NDItemId, typename _Size, typename _TransformPattern, typename _ReducePattern,
          typename _AccLocal, typename _Temp, typename... _Acc>
void
__device_reduce_kernel(const _NDItemId __item_id, const _Size __n, const _Size __n_items,
                       _TransformPattern __transform_pattern, _ReducePattern __reduce_pattern,
                       const _AccLocal& __local_mem, const _Temp& __temp_acc, const _Acc&... __acc)
{
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_idx = __item_id.get_group(0);
    // 1. Initialization (transform part). Fill local memory
    __transform_pattern(__item_id, __n, /*global_offset*/ (_Size)0, __local_mem, __acc...);
    __dpl_sycl::__group_barrier(__item_id);
    // 2. Reduce within work group using local memory
    _Tp __result = __reduce_pattern(__item_id, __n_items, __local_mem);
    if (__local_idx == 0)
        __temp_acc[__group_idx] = __result;
}

//------------------------------------------------------------------------
// parallel_transform_reduce - async patterns
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Parallel_transform_reduce for a small arrays using a single work group.
// Transforms and reduces __work_group_size * __iters_per_work_item elements.
template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _KernelName>
struct __parallel_transform_reduce_small_submitter;

template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename... _Name>
struct __parallel_transform_reduce_small_submitter<_Tp, __work_group_size, __iters_per_work_item,
                                                   __internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType, typename _Res,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Res& __res, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        const _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        return __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            // get an access to data under SYCL buffer
            oneapi::dpl::__ranges::__require_access(__cgh, __res, __rngs...);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                    __init, __temp_local, __res, __rngs...);
                });
        });
    }
}; // struct __parallel_transform_reduce_small_submitter

template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item,
          typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType, typename _Res,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_impl(_ExecutionPolicy&& __exec, const _Size __n, _ReduceOp __reduce_op,
                                       _TransformOp __transform_op, _InitType __init, _Res& __res, _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_small_kernel<::std::integral_constant<::std::uint8_t, __iters_per_work_item>, _CustomName, _Res>>;

    return __parallel_transform_reduce_small_submitter<_Tp, __work_group_size, __iters_per_work_item, _ReduceKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
        ::std::forward<_Ranges>(__rngs)...);
}

// Submits the first kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses multiple work groups that each reduce __work_group_size * __iters_per_work_item items and store the preliminary
// results in __temp.
template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter;

template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item,
          typename... _KernelName>
struct __parallel_transform_reduce_device_kernel_submitter<_Tp, __work_group_size, __iters_per_work_item,
                                                           __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              typename _Temp, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    auto
    operator()(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op, _TransformOp __transform_op,
               _InitType __init, _Temp& __temp, _Ranges&&... __rngs) const
    {
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // number of buffer elements processed within workgroup
        constexpr _Size __size_per_work_group = __iters_per_work_item * __work_group_size;
        const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        return __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            // get an access to data under SYCL buffer
            oneapi::dpl::__ranges::__require_access(__cgh, __temp, __rngs...);
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item<1> __item_id) {
                    __device_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                __temp_local, __temp, __rngs...);
                });
        });
    }
}; // struct __parallel_transform_reduce_device_kernel_submitter

// Submits the second kernel of the parallel_transform_reduce for mid-sized arrays.
// Uses a single work groups to reduce __n preliminary results stored in __temp and returns a future object with the
// result buffer.
template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item, typename _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter;

template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item,
          typename... _KernelName>
struct __parallel_transform_reduce_work_group_kernel_submitter<_Tp, __work_group_size, __iters_per_work_item,
                                                               __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              typename _Res, typename _Temp,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    auto
    operator()(_ExecutionPolicy&& __exec, sycl::event& __reduce_event, _Size __n, _ReduceOp __reduce_op,
               _TransformOp __transform_op, _InitType __init, _Res& __res, _Temp& __temp) const
    {
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        auto __transform_pattern =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _NoOpFunctor>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

        // Lower the work group size of the second kernel to the next power of 2 if __n < __work_group_size.
        auto __work_group_size2 = __work_group_size;
        if constexpr (__iters_per_work_item == 1)
        {
            if (__n < __work_group_size)
            {
                __work_group_size2 = __n;
                if ((__work_group_size2 & (__work_group_size2 - 1)) != 0)
                    __work_group_size2 = oneapi::dpl::__internal::__dpl_bit_floor(__work_group_size2) << 1;
            }
        }
        const _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        return __exec.queue().submit([&, __n, __n_items](sycl::handler& __cgh) {
            __cgh.depends_on(__reduce_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __temp, __res); // get an access to data under SYCL buffer
            __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size2), __cgh);

            __cgh.parallel_for<_KernelName...>(
                sycl::nd_range<1>(sycl::range<1>(__work_group_size2), sycl::range<1>(__work_group_size2)),
                [=](sycl::nd_item<1> __item_id) {
                    __work_group_reduce_kernel<_Tp>(__item_id, __n, __n_items, __transform_pattern, __reduce_pattern,
                                                    __init, __temp_local, __res, __temp);
                });
        });
    }
}; // struct __parallel_transform_reduce_work_group_kernel_submitter

template <typename _Tp, ::std::uint16_t __work_group_size, ::std::uint8_t __iters_per_work_item_device_kernel,
          ::std::uint8_t __iters_per_work_item_work_group_kernel, typename _ExecutionPolicy, typename _Size,
          typename _ReduceOp, typename _TransformOp, typename _InitType, typename _Res,
          typename _Temp, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
auto
__parallel_transform_reduce_mid_impl(_ExecutionPolicy&& __exec, _Size __n, _ReduceOp __reduce_op,
                                     _TransformOp __transform_op, _InitType __init, _Res& __res, _Temp& __temp,
                                     _Ranges&&... __rngs)
{
    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;

    // The same value for __iters_per_work_item_device_kernel is currently used. Include
    // __iters_per_work_item_device_kernel in case this changes in the future.
    using _ReduceDeviceKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_device_kernel<_CustomName>>;
    using _ReduceWorkGroupKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__reduce_mid_work_group_kernel<
            ::std::integral_constant<::std::uint8_t, __iters_per_work_item_work_group_kernel>, _CustomName>>;

    // number of buffer elements processed within workgroup
    constexpr _Size __size_per_work_group = __iters_per_work_item_device_kernel * __work_group_size;
    const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);

    sycl::event __reduce_event =
        __parallel_transform_reduce_device_kernel_submitter<_Tp, __work_group_size, __iters_per_work_item_device_kernel,
                                                            _ReduceDeviceKernel>()(
            __exec, __n, __reduce_op, __transform_op, __init, __temp, ::std::forward<_Ranges>(__rngs)...);

    __n = __n_groups; // Number of preliminary results from the device kernel.
    return __parallel_transform_reduce_work_group_kernel_submitter<
        _Tp, __work_group_size, __iters_per_work_item_work_group_kernel, _ReduceWorkGroupKernel>()(
        ::std::forward<_ExecutionPolicy>(__exec), __reduce_event, __n, __reduce_op, __transform_op, __init, __res,
        __temp);
}

// General implementation using a tree reduction
template <typename _Tp, ::std::uint8_t __iters_per_work_item>
struct __parallel_transform_reduce_impl
{
    template <typename _ExecutionPolicy, typename _Size, typename _ReduceOp, typename _TransformOp, typename _InitType,
              typename _Res, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
              typename... _Ranges>
    static auto
    submit(_ExecutionPolicy&& __exec, _Size __n, ::std::uint16_t __work_group_size, _ReduceOp __reduce_op,
           _TransformOp __transform_op, _InitType __init, _Res& __res, _Ranges&&... __rngs)
    {
        using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
        using _CustomName = typename _Policy::kernel_name;
        using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;
        using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __reduce_kernel, _CustomName, _ReduceOp, _TransformOp, _NoOpFunctor, _Ranges...>;

        auto __transform_pattern1 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _TransformOp>{
                __reduce_op, __transform_op};
        auto __transform_pattern2 =
            unseq_backend::transform_reduce<_ExecutionPolicy, __iters_per_work_item, _ReduceOp, _NoOpFunctor>{
                __reduce_op, _NoOpFunctor{}};
        auto __reduce_pattern = unseq_backend::reduce_over_group<_ExecutionPolicy, _ReduceOp, _Tp>{__reduce_op};

#if _ONEDPL_COMPILE_KERNEL
        auto __kernel = __internal::__kernel_compiler<_ReduceKernel>::__compile(__exec);
        __work_group_size = ::std::min(
            __work_group_size, (::std::uint16_t)oneapi::dpl::__internal::__kernel_work_group_size(__exec, __kernel));
#endif

        _Size __size_per_work_group =
            __iters_per_work_item * __work_group_size; // number of buffer elements processed within workgroup
        _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        _Size __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);

        // Create temporary global buffers to store temporary values
        sycl::buffer<_Tp> __temp(sycl::range<1>(2 * __n_groups));
        // __is_first == true. Reduce over each work_group
        // __is_first == false. Reduce between work groups
        bool __is_first = true;

        // For memory utilization it's better to use one big buffer instead of two small because size of the buffer is
        // close to a few MB
        _Size __offset_1 = 0;
        _Size __offset_2 = __n_groups;

        sycl::event __reduce_event;
        do
        {
            __reduce_event = __exec.queue().submit([&, __is_first, __offset_1, __offset_2, __n, __n_items,
                                                    __n_groups](sycl::handler& __cgh) {
                __cgh.depends_on(__reduce_event);

                // get an access to data under SYCL buffer
                oneapi::dpl::__ranges::__require_access(__cgh, __res, __rngs...);
                sycl::accessor __temp_acc{__temp, __cgh, sycl::read_write};
                __dpl_sycl::__local_accessor<_Tp> __temp_local(sycl::range<1>(__work_group_size), __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__kernel.get_kernel_bundle());
#endif
                __cgh.parallel_for<_ReduceKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_KERNEL_BUNDLE_PRESENT
                    __kernel,
#endif
                    sycl::nd_range<1>(sycl::range<1>(__n_groups * __work_group_size),
                                      sycl::range<1>(__work_group_size)),
                    [=](sycl::nd_item<1> __item_id) {
                        auto __local_idx = __item_id.get_local_id(0);
                        auto __group_idx = __item_id.get_group(0);
                        // 1. Initialization (transform part). Fill local memory
                        if (__is_first)
                            __transform_pattern1(__item_id, __n, /*global_offset*/ (_Size)0, __temp_local, __rngs...);
                        else
                            __transform_pattern2(__item_id, __n, __offset_2, __temp_local, __temp_acc);
                        __dpl_sycl::__group_barrier(__item_id);
                        // 2. Reduce within work group using local memory
                        _Tp __result = __reduce_pattern(__item_id, __n_items, __temp_local);
                        if (__local_idx == 0)
                        {
                            // final reduction
                            if (__n_groups == 1)
                            {
                                __reduce_pattern.apply_init(__init, __result);
                                __res[0] = __result;
                            }

                            __temp_acc[__offset_1 + __group_idx] = __result;
                        }
                    });
            });
            if (__is_first)
                __is_first = false;
            ::std::swap(__offset_1, __offset_2);
            __n = __n_groups;
            __n_items = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);
            __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_work_group);
        } while (__n > 1);

        return __reduce_event;
    }
}; // struct __parallel_transform_reduce_impl

// Use a single work group to reduce small arrays (__work_group_size * __iters_per_work_item).
template <typename _Tp, typename _Commutative, typename _ExecutionPolicy, typename _ReduceOp, typename _TransformOp,
          typename _InitType, typename _Res,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_small_caller(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                                         _InitType __init, _Res& __res, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);
    assert(__n <= __reduce_max_n_small);

    if (__n <= 256)
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 1>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 512)
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 2>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 1024)
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 4>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 2048)
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 8>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 4096)
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 16>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else
    {
        return __parallel_transform_reduce_small_impl<_Tp, __reduce_work_group_size, 32>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res,
            ::std::forward<_Ranges>(__rngs)...);
    }
}

// Use two-step tree reduction.
// First step reduces __work_group_size * __iters_per_work_item_device_kernel elements.
// Second step reduces __work_group_size * __iters_per_work_item_work_group_kernel elements.
template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _Commutative, typename _ExecutionPolicy,
          typename _InitType, typename _Res, typename _Temp,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0, typename... _Ranges>
auto
__parallel_transform_reduce_mid_caller(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                                       _InitType __init, _Res& __res, _Temp& __temp, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);
    assert(__n <= __reduce_max_n_mid);

    if (__n <= 2097152)
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 1>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 4194304)
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 2>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 8388608)
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 4>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 16777216)
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 8>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__n <= 33554432)
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 16>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
    else
    {
        return __parallel_transform_reduce_mid_impl<_Tp, __reduce_work_group_size, __reduce_iters_per_work_item, 32>(
            ::std::forward<_ExecutionPolicy>(__exec), __n, __reduce_op, __transform_op, __init, __res, __temp,
            ::std::forward<_Ranges>(__rngs)...);
    }
}

// Use a wrapped policy for the buffer implementation
template <typename _Name>
class __reduce_wrapper
{
};

// Synchronous pattern. The general implementation uses buffers. Using USM memory can be beneficial since it reduces
// the overheads of managing migratable memory. USM host memory is further beneficial on GPUs utilizing L0. Other
// runtimes show significant overheads for pinning the host memory.
//
// The binary operator must be associative but commutativity is only required by some of the algorithms using
// __parallel_transform_reduce. This is provided by the _Commutative parameter. The current implementation uses a
// generic implementation that processes elements in order. However, future improvements might be possible utilizing
// the commutative property of the respective algorithms.
//
// A single-work group implementation is used for small arrays.
// Mid-sized arrays use two tree reductions with independent __iters_per_work_item.
// Big arrays are processed with a recursive tree reduction. __work_group_size * __iters_per_work_item elements are
// reduced in each step.
template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _Commutative, typename _ExecutionPolicy,
          typename _InitType, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
auto
__parallel_transform_reduce_sync(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                                 _InitType __init, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);
    using _Size = typename ::std::decay<decltype(__n)>::type;

    auto __queue = __exec.queue();
    auto __device = __queue.get_device();

    // Get the work group size adjusted to the local memory limit.
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: find a way to generalize getting of reliable work-group size.
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_Tp) * 2);

    // Use USM device memory if supported or USM host memory if executed on a L0 GPU.
    if (__device.has(sycl::aspect::usm_device_allocations))
    {
        sycl::usm::alloc __usm_kind = sycl::usm::alloc::device;
        bool __supports_usm_host = __device.has(sycl::aspect::usm_host_allocations);
        bool __is_gpu = __device.is_gpu();
        bool __is_level_zero = __device.get_backend() == sycl::backend::ext_oneapi_level_zero;
        bool __use_usm_host = (__supports_usm_host && __is_gpu && __is_level_zero);
        if (__use_usm_host)
            __usm_kind = sycl::usm::alloc::host;
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _Tp, _Tp*> __res_usm(__exec, 1,
                                                                                                       __usm_kind);
        auto __keep = oneapi::dpl::__ranges::__get_sycl_range<
            __par_backend_hetero::access_mode::write,
            oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _Tp, _Tp*>>();
        auto __buf_temp = __keep(__res_usm.get(), __res_usm.get() + 1);
        auto __res = __buf_temp.all_view();
        if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_small)
        {
            __parallel_transform_reduce_small_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
                ::std::forward<_ExecutionPolicy>(__exec), __reduce_op, __transform_op, __init, __res,
                ::std::forward<_Ranges>(__rngs)...)
                .wait();
        }
        else if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_mid)
        {
            const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __reduce_max_n_small);
            oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _Tp, _Tp*> __temp_usm(
                __exec, __n_groups, sycl::usm::alloc::device);
            auto __keep_temp = oneapi::dpl::__ranges::__get_sycl_range<
                __par_backend_hetero::access_mode::read_write,
                oneapi::dpl::__par_backend_hetero::__internal::__buffer<_ExecutionPolicy, _Tp, _Tp*>>();
            auto __buf_temp = __keep_temp(__temp_usm.get(), __temp_usm.get() + 1);
            auto __temp = __buf_temp.all_view();
            __parallel_transform_reduce_mid_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
                ::std::forward<_ExecutionPolicy>(__exec), __reduce_op, __transform_op, __init, __res, __temp,
                ::std::forward<_Ranges>(__rngs)...)
                .wait();
        }
        else
        {
            __parallel_transform_reduce_impl<_Tp, __reduce_iters_per_work_item>::submit(
                ::std::forward<_ExecutionPolicy>(__exec), __n, __work_group_size, __reduce_op, __transform_op, __init,
                __res, ::std::forward<_Ranges>(__rngs)...)
                .wait();
        }
        if (__use_usm_host)
            return __res_usm.get()[0];
        else
        {
            _Tp __host_res[1];
            __queue.copy(__res_usm.get(), __host_res, 1).wait();
            return __host_res[0];
        }
    }
    // Use the general buffer implementation if the device doesn't support USM memory.
    else
    {

        sycl::buffer<_Tp> __res_buffer(sycl::range<1>(1));
        auto __res = oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::write>(__res_buffer);
        if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_small)
        {
            __parallel_transform_reduce_small_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
                oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                    ::std::forward<_ExecutionPolicy>(__exec)),
                __reduce_op, __transform_op, __init, __res, ::std::forward<_Ranges>(__rngs)...);
        }
        else if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_mid)
        {
            const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __reduce_max_n_small);
            sycl::buffer<_Tp> __temp_buffer((sycl::range<1>(__n_groups)));
            auto __temp =
                oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::read_write>(__temp_buffer);
            __parallel_transform_reduce_mid_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
                oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                    ::std::forward<_ExecutionPolicy>(__exec)),
                __reduce_op, __transform_op, __init, __res, __temp, ::std::forward<_Ranges>(__rngs)...);
        }
        else
        {
            __parallel_transform_reduce_impl<_Tp, __reduce_iters_per_work_item>::submit(
                oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                    ::std::forward<_ExecutionPolicy>(__exec)),
                __n, __work_group_size, __reduce_op, __transform_op, __init, __res, ::std::forward<_Ranges>(__rngs)...);
        }

        return __res_buffer.get_host_access(sycl::read_only)[0];
    }
}

// Asynchronous pattern. Returns a future-like object based on a result buffer.
template <typename _Tp, typename _ReduceOp, typename _TransformOp, typename _Commutative, typename _ExecutionPolicy,
          typename _InitType, oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0,
          typename... _Ranges>
auto
__parallel_transform_reduce_async(_ExecutionPolicy&& __exec, _ReduceOp __reduce_op, _TransformOp __transform_op,
                                  _InitType __init, _Ranges&&... __rngs)
{
    auto __n = oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    assert(__n > 0);
    using _Size = typename ::std::decay<decltype(__n)>::type;

    // Get the work group size adjusted to the local memory limit.
    // Pessimistically double the memory requirement to take into account memory used by compiled kernel.
    // TODO: find a way to generalize getting of reliable work-group size.
    ::std::size_t __work_group_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__exec, sizeof(_Tp) * 2);

    sycl::event __reduce_event;
    sycl::buffer<_Tp> __res_buffer(sycl::range<1>(1));
    auto __res = oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::write>(__res_buffer);
    if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_small)
    {
        __reduce_event = __parallel_transform_reduce_small_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __reduce_op, __transform_op, __init, __res, ::std::forward<_Ranges>(__rngs)...);
    }
    else if (__work_group_size >= __reduce_work_group_size && __n <= __reduce_max_n_mid)
    {
        const _Size __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __reduce_max_n_small);
        sycl::buffer<_Tp> __temp_buffer((sycl::range<1>(__n_groups)));
        auto __temp =
            oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::read_write>(__temp_buffer);
        __reduce_event = __parallel_transform_reduce_mid_caller<_Tp, _ReduceOp, _TransformOp, _Commutative>(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __reduce_op, __transform_op, __init, __res, __temp, ::std::forward<_Ranges>(__rngs)...);
    }
    else
    {
        __reduce_event = __parallel_transform_reduce_impl<_Tp, __reduce_iters_per_work_item>::submit(
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __n, __work_group_size, __reduce_op, __transform_op, __init, __res, ::std::forward<_Ranges>(__rngs)...);
    }

    return __future(__reduce_event, __res_buffer);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_H
