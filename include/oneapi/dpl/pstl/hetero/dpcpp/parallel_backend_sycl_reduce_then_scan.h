// -*- C++ -*-
//===-- parallel_backend_sycl_reduce_then_scan.h ---------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _SubGroup, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__exclusive_sub_group_masked_scan(const _SubGroup& __sub_group, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                                  _ValueType& __value, _BinaryOp __binary_op, _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        auto __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    _LazyValueType __old_init;
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        if (__sub_group_local_id == 0)
            __old_init.__setup(__init_and_carry.__v);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }

    __value = sycl::shift_group_right(__sub_group, __value, 1);
    if constexpr (__init_present)
    {
        if (__sub_group_local_id == 0)
        {
            __value = __old_init.__v;
            __old_init.__destroy();
        }
    }
    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __init_present, typename _MaskOp, typename _InitBroadcastId,
          typename _SubGroup, typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__inclusive_sub_group_masked_scan(const _SubGroup& __sub_group, _MaskOp __mask_fn, _InitBroadcastId __init_broadcast_id,
                                  _ValueType& __value, _BinaryOp __binary_op, _LazyValueType& __init_and_carry)
{
    std::uint8_t __sub_group_local_id = __sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t __shift = 1; __shift <= __sub_group_size / 2; __shift <<= 1)
    {
        auto __partial_carry_in = sycl::shift_group_right(__sub_group, __value, __shift);
        if (__mask_fn(__sub_group_local_id, __shift))
        {
            __value = __binary_op(__partial_carry_in, __value);
        }
    }
    if constexpr (__init_present)
    {
        __value = __binary_op(__init_and_carry.__v, __value);
        __init_and_carry.__v = sycl::group_broadcast(__sub_group, __value, __init_broadcast_id);
    }
    else
    {
        __init_and_carry.__setup(sycl::group_broadcast(__sub_group, __value, __init_broadcast_id));
    }

    //return by reference __value and __init_and_carry
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _SubGroup,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType>
void
__sub_group_scan(const _SubGroup& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                 _LazyValueType& __init_and_carry)
{
    auto __mask_fn = [](auto __sub_group_local_id, auto __offset) { return __sub_group_local_id >= __offset; };
    constexpr auto __init_broadcast_id = __sub_group_size - 1;
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    else
    {
        __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, typename _SubGroup,
          typename _BinaryOp, typename _ValueType, typename _LazyValueType, typename _SizeType>
void
__sub_group_scan_partial(const _SubGroup& __sub_group, _ValueType& __value, _BinaryOp __binary_op,
                         _LazyValueType& __init_and_carry, _SizeType __elements_to_process)
{
    auto __mask_fn = [__elements_to_process](auto __sub_group_local_id, auto __offset) {
        return __sub_group_local_id >= __offset && __sub_group_local_id < __elements_to_process;
    };
    auto __init_broadcast_id = __elements_to_process - 1;
    if constexpr (__is_inclusive)
    {
        __inclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
    else
    {
        __exclusive_sub_group_masked_scan<__sub_group_size, __init_present>(__sub_group, __mask_fn, __init_broadcast_id,
                                                                            __value, __binary_op, __init_and_carry);
    }
}

template <std::uint8_t __sub_group_size, bool __is_inclusive, bool __init_present, bool __capture_output,
          std::uint32_t __max_inputs_per_item, typename _SubGroup, typename _UnaryOp, typename _BinaryOp,
          typename _LazyValueType, typename _InRng, typename _OutRng>
void
__scan_through_elements_helper(const _SubGroup& __sub_group, _UnaryOp __unary_op, _BinaryOp __binary_op,
                               _LazyValueType& __sub_group_carry, _InRng __in_rng, _OutRng __out_rng,
                               std::size_t __start_idx, std::size_t __n, std::uint32_t __iters_per_item,
                               std::size_t __subgroup_start_idx, std::uint32_t __sub_group_id,
                               std::uint32_t __active_subgroups)
{
    bool __is_full_block = (__iters_per_item == __max_inputs_per_item);
    bool __is_full_thread = __subgroup_start_idx + __iters_per_item * __sub_group_size <= __n;
    if (__is_full_thread && __is_full_block)
    {
        auto __v = __unary_op(__in_rng[__start_idx]);
        __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(__sub_group, __v, __binary_op,
                                                                           __sub_group_carry);
        if constexpr (__capture_output)
        {
            __out_rng[__start_idx] = __v;
        }

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __j = 1; __j < __max_inputs_per_item; __j++)
        {
            __v = __unary_op(__in_rng[__start_idx + __j * __sub_group_size]);
            __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(__sub_group, __v, __binary_op,
                                                                                        __sub_group_carry);
            if constexpr (__capture_output)
            {
                __out_rng[__start_idx + __j * __sub_group_size] = __v;
            }
        }
    }
    else if (__is_full_thread)
    {
        auto __v = __unary_op(__in_rng[__start_idx]);
        __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(__sub_group, __v, __binary_op,
                                                                           __sub_group_carry);
        if constexpr (__capture_output)
        {
            __out_rng[__start_idx] = __v;
        }
        for (std::uint32_t __j = 1; __j < __iters_per_item; __j++)
        {
            __v = __unary_op(__in_rng[__start_idx + __j * __sub_group_size]);
            __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(__sub_group, __v, __binary_op,
                                                                                        __sub_group_carry);
            if constexpr (__capture_output)
            {
                __out_rng[__start_idx + __j * __sub_group_size] = __v;
            }
        }
    }
    else
    {
        if (__sub_group_id < __active_subgroups)
        {
            auto __iters = oneapi::dpl::__internal::__dpl_ceiling_div(__n - __subgroup_start_idx, __sub_group_size);

            if (__iters == 1)
            {
                auto __v = __unary_op(__in_rng[__start_idx]);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, __init_present>(
                    __sub_group, __v, __binary_op, __sub_group_carry, __n - __subgroup_start_idx);
                if constexpr (__capture_output)
                {
                    if (__start_idx < __n)
                        __out_rng[__start_idx] = __v;
                }
            }
            else
            {
                auto __v = __unary_op(__in_rng[__start_idx]);
                __sub_group_scan<__sub_group_size, __is_inclusive, __init_present>(__sub_group, __v, __binary_op,
                                                                                   __sub_group_carry);
                if constexpr (__capture_output)
                {
                    __out_rng[__start_idx] = __v;
                }

                for (int __j = 1; __j < __iters - 1; __j++)
                {
                    auto __local_idx = __start_idx + __j * __sub_group_size;
                    __v = __unary_op(__in_rng[__local_idx]);
                    __sub_group_scan<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                        __sub_group, __v, __binary_op, __sub_group_carry);
                    if constexpr (__capture_output)
                    {
                        __out_rng[__local_idx] = __v;
                    }
                }

                auto __offset = __start_idx + (__iters - 1) * __sub_group_size;
                auto __local_idx = (__offset < __n) ? __offset : __n - 1;
                __v = __unary_op(__in_rng[__local_idx]);
                __sub_group_scan_partial<__sub_group_size, __is_inclusive, /*__init_present=*/true>(
                    __sub_group, __v, __binary_op, __sub_group_carry,
                    __n - (__subgroup_start_idx + (__iters - 1) * __sub_group_size));
                if constexpr (__capture_output)
                {
                    if (__offset < __n)
                        __out_rng[__offset] = __v;
                }
            }
        }
    }
}

template <typename... _Name>
class __reduce_then_scan_reduce_kernel;

template <typename... _Name>
class __reduce_then_scan_scan_kernel;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __is_inclusive,
          typename _BinaryOperation, typename _UnaryOperation, typename _InitType, typename _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __is_inclusive,
          typename _BinaryOperation, typename _UnaryOperation, typename _InitType, typename... _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter<__sub_group_size, __max_inputs_per_item, __is_inclusive,
                                                    _BinaryOperation, _UnaryOperation, _InitType,
                                                    __internal::__optional_kernel_name<_KernelName...>>
{
    // Step 1 - SubGroupReduce is expected to perform sub-group reductions to global memory
    // input buffer
    template <typename _ExecutionPolicy, typename _InRng, typename _TmpStorageAcc>
    auto
    operator()(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _TmpStorageAcc __tmp_storage,
               const sycl::event& __prior_event, const std::size_t __inputs_per_sub_group,
               const std::size_t __inputs_per_item, const std::size_t __block_num, const bool __is_full_block) const
    {
        using _InValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        return __exec.queue().submit([&, this](sycl::handler& __cgh) {
            sycl::local_accessor<_InValueType> __sub_group_partials(__num_sub_groups_local, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng);
            __cgh.parallel_for<_KernelName...>(__nd_range, [=,
                                                            *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(
                                                               __sub_group_size)]] {
                auto __id = __ndi.get_global_id(0);
                auto __lid = __ndi.get_local_id(0);
                auto __g = __ndi.get_group(0);
                auto __sub_group = __ndi.get_sub_group();
                auto __sub_group_id = __sub_group.get_group_linear_id();
                auto __sub_group_local_id = __sub_group.get_local_linear_id();

                oneapi::dpl::__internal::__lazy_ctor_storage<_InValueType> __sub_group_carry;
                std::size_t __group_start_idx =
                    (__block_num * __max_block_size) + (__g * __inputs_per_sub_group * __num_sub_groups_local);
                if (__n <= __group_start_idx)
                    return; // exit early for empty groups (TODO: avoid launching these?)

                std::size_t __elements_in_group =
                    std::min(__n - __group_start_idx, std::size_t(__num_sub_groups_local * __inputs_per_sub_group));
                std::uint32_t __active_subgroups =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__elements_in_group, __inputs_per_sub_group);
                std::size_t __subgroup_start_idx = __group_start_idx + (__sub_group_id * __inputs_per_sub_group);

                std::size_t __start_idx = __subgroup_start_idx + __sub_group_local_id;

                if (__sub_group_id < __active_subgroups)
                {
                    // adjust for lane-id
                    // compute sub-group local pfix on T0..63, K samples/T, send to accumulator kernel
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/false, __max_inputs_per_item>(
                        __sub_group, __unary_op, __binary_op, __sub_group_carry, __in_rng, nullptr, __start_idx, __n,
                        __inputs_per_item, __subgroup_start_idx, __sub_group_id, __active_subgroups);
                    if (__sub_group_local_id == 0)
                        __sub_group_partials[__sub_group_id] = __sub_group_carry.__v;
                    __sub_group_carry.__destroy();
                }
                // TODO: This is slower then ndi.barrier which was removed in SYCL2020. Can we do anything about it?
                //sycl::group_barrier(ndi.get_group());
                __ndi.barrier(sycl::access::fence_space::local_space);

                // compute sub-group local prefix sums on (T0..63) carries
                // and store to scratch space at the end of dst; next
                // accumulator kernel takes M thread carries from scratch
                // to compute a prefix sum on global carries
                if (__sub_group_id == 0)
                {
                    __start_idx = (__g * __num_sub_groups_local);
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    if (__iters == 1)
                    {
                        auto __load_idx = (__sub_group_local_id < __active_subgroups)
                                              ? __sub_group_local_id
                                              : (__active_subgroups - 1); // else is unused dummy value
                        auto __v = __sub_group_partials[__load_idx];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __binary_op, __sub_group_carry,
                            __active_subgroups - __subgroup_start_idx);
                        if (__sub_group_local_id < __active_subgroups)
                            __tmp_storage[__start_idx + __sub_group_local_id] = __v;
                    }
                    else
                    {
                        //need to pull out first iteration tp avoid identity
                        auto __v = __sub_group_partials[__sub_group_local_id];
                        __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                            __sub_group, __v, __binary_op, __sub_group_carry);
                        __tmp_storage[__start_idx + __sub_group_local_id] = __v;

                        for (int __i = 1; __i < __iters - 1; __i++)
                        {
                            __v = __sub_group_partials[__i * __sub_group_size + __sub_group_local_id];
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                __sub_group, __v, __binary_op, __sub_group_carry);
                            __tmp_storage[__start_idx + __i * __sub_group_size + __sub_group_local_id] = __v;
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.

                        // else is an unused dummy value
                        auto __proposed_idx = (__iters - 1) * __sub_group_size + __sub_group_local_id;
                        auto __load_idx =
                            (__proposed_idx < __num_sub_groups_local) ? __proposed_idx : (__num_sub_groups_local - 1);

                        __v = __sub_group_partials[__load_idx];
                        __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                            __sub_group, __v, __binary_op, __sub_group_carry, __num_sub_groups_local);
                        if (__proposed_idx < __num_sub_groups_local)
                            __tmp_storage[__start_idx + __proposed_idx] = __v;
                    }

                    __sub_group_carry.__destroy();
                }
            });
        });
    }

    // Constant parameters throughout all blocks
    const sycl::nd_range<1> __nd_range;

    const std::size_t __max_block_size;
    const std::size_t __num_sub_groups_local;
    const std::size_t __num_sub_groups_global;
    const std::size_t __num_work_items;
    const std::size_t __n;

    const _BinaryOperation __binary_op;
    const _UnaryOperation __unary_op;
    _InitType __init;

    // TODO: Add the mask functors here to generalize for scan-based algorithms
};

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __is_inclusive,
          typename _BinaryOperation, typename _UnaryOperation, typename _InitType, typename _KernelName>
struct __parallel_reduce_then_scan_scan_submitter;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __is_inclusive,
          typename _BinaryOperation, typename _UnaryOperation, typename _InitType, typename... _KernelName>
struct __parallel_reduce_then_scan_scan_submitter<__sub_group_size, __max_inputs_per_item, __is_inclusive,
                                                  _BinaryOperation, _UnaryOperation, _InitType,
                                                  __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _TmpStorageAcc>
    auto
    operator()(_ExecutionPolicy&& __exec, _InRng&& __in_rng, _OutRng&& __out_rng, _TmpStorageAcc __tmp_storage,
               const sycl::event& __prior_event, const std::size_t __inputs_per_sub_group,
               const std::size_t __inputs_per_item, const std::size_t __block_num, const bool __is_full_block) const
    {
        using _InValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        using _InitValueType = typename _InitType::__value_type;
        return __exec.queue().submit([&, this](sycl::handler& __cgh) {
            sycl::local_accessor<_InValueType> __sub_group_partials(__num_sub_groups_local + 1, __cgh);
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_rng, __out_rng);
            __cgh.parallel_for<_KernelName...>(__nd_range, [=,
                                                            *this](sycl::nd_item<1> __ndi) [[sycl::reqd_sub_group_size(
                                                               __sub_group_size)]] {
                auto __id = __ndi.get_global_id(0);
                auto __lid = __ndi.get_local_id(0);
                auto __g = __ndi.get_group(0);
                auto __sub_group = __ndi.get_sub_group();
                auto __sub_group_id = __sub_group.get_group_linear_id();
                auto __sub_group_local_id = __sub_group.get_local_linear_id();

                auto __group_start_idx =
                    (__block_num * __max_block_size) + (__g * __inputs_per_sub_group * __num_sub_groups_local);
                if (__n <= __group_start_idx)
                    return; // exit early for empty groups (TODO: avoid launching these?)

                std::size_t __elements_in_group =
                    std::min(__n - __group_start_idx, std::size_t(__num_sub_groups_local * __inputs_per_sub_group));
                std::uint32_t __active_subgroups =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__elements_in_group, __inputs_per_sub_group);
                oneapi::dpl::__internal::__lazy_ctor_storage<_InValueType> __carry_last;
                oneapi::dpl::__internal::__lazy_ctor_storage<_InValueType> __value;

                // propogate carry in from previous block
                oneapi::dpl::__internal::__lazy_ctor_storage<_InValueType> __sub_group_carry;

                // on the first sub-group in a work-group (assuming S subgroups in a work-group):
                // 1. load S sub-group local carry pfix sums (T0..TS-1) to slm
                // 2. load 32, 64, 96, etc. TS-1 work-group carry-outs (32 for WG num<32, 64 for WG num<64, etc.),
                //    and then compute the prefix sum to generate global carry out
                //    for each WG, i.e., prefix sum on TS-1 carries over all WG.
                // 3. on each WG select the adjacent neighboring WG carry in
                // 4. on each WG add the global carry-in to S sub-group local pfix sums to
                //    get a T-local global carry in
                // 5. recompute T-local pfix values, add the T-local global carries,
                //    and then write back the final values to memory
                if (__sub_group_id == 0)
                {
                    // step 1) load to Xe slm the WG-local S prefix sums
                    //         on WG T-local carries
                    //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
                    //           S: sum(T0 carry...TS carry)
                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);
                    auto __subgroups_before_my_group = __g * __num_sub_groups_local;
                    std::uint8_t __i = 0;
                    for (; __i < __iters - 1; __i++)
                    {
                        __sub_group_partials[__i * __sub_group_size + __sub_group_local_id] =
                            __tmp_storage[__subgroups_before_my_group + __i * __sub_group_size + __sub_group_local_id];
                    }
                    if (__i * __sub_group_size + __sub_group_local_id < __active_subgroups)
                    {
                        __sub_group_partials[__i * __sub_group_size + __sub_group_local_id] =
                            __tmp_storage[__subgroups_before_my_group + __i * __sub_group_size + __sub_group_local_id];
                    }

                    // step 2) load 32, 64, 96, etc. work-group carry outs on every work-group; then
                    //         compute the prefix in a sub-group to get global work-group carries
                    //         memory accesses: gather(63, 127, 191, 255, ...)
                    std::uint32_t __offset = __num_sub_groups_local - 1;
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    if (__g > 0)
                    {
                        // only need the last element from each scan of num_sub_groups_local subgroup reductions
                        const auto __elements_to_process = __subgroups_before_my_group / __num_sub_groups_local;
                        const auto __pre_carry_iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(__elements_to_process, __sub_group_size);
                        if (__pre_carry_iters == 1)
                        {
                            // single partial scan
                            auto __proposed_idx = __num_sub_groups_local * __sub_group_local_id + __offset;
                            auto __remaining_elements = __elements_to_process;
                            auto __reduction_idx = (__proposed_idx < __subgroups_before_my_group)
                                                       ? __proposed_idx
                                                       : __subgroups_before_my_group - 1;
                            __value.__setup(__tmp_storage[__reduction_idx]);
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/false>(__sub_group, __value.__v, __binary_op,
                                                                               __carry_last, __remaining_elements);
                        }
                        else
                        {
                            // multiple iterations
                            // first 1 full
                            __value.__setup(__tmp_storage[__num_sub_groups_local * __sub_group_local_id + __offset]);
                            __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/false>(
                                __sub_group, __value.__v, __binary_op, __carry_last);

                            // then some number of full iterations
                            for (int __i = 1; __i < __pre_carry_iters - 1; __i++)
                            {
                                auto __reduction_idx = __i * __num_sub_groups_local * __sub_group_size +
                                                       __num_sub_groups_local * __sub_group_local_id + __offset;
                                __value.__v = __tmp_storage[__reduction_idx];
                                __sub_group_scan<__sub_group_size, /*__is_inclusive=*/true, /*__init_present=*/true>(
                                    __sub_group, __value.__v, __binary_op, __carry_last);
                            }

                            // final partial iteration
                            auto __proposed_idx = (__pre_carry_iters - 1) * __num_sub_groups_local * __sub_group_size +
                                                  __num_sub_groups_local * __sub_group_local_id + __offset;
                            auto __remaining_elements =
                                __elements_to_process - ((__pre_carry_iters - 1) * __sub_group_size);
                            auto __reduction_idx = (__proposed_idx < __subgroups_before_my_group)
                                                       ? __proposed_idx
                                                       : __subgroups_before_my_group - 1;
                            __value.__v = __tmp_storage[__reduction_idx];
                            __sub_group_scan_partial<__sub_group_size, /*__is_inclusive=*/true,
                                                     /*__init_present=*/true>(__sub_group, __value.__v, __binary_op,
                                                                              __carry_last, __remaining_elements);
                        }
                    }
                }

                // N.B. barrier could be earlier, guarantees slm local carry update
                //sycl::group_barrier(ndi.get_group());
                __ndi.barrier(sycl::access::fence_space::local_space);

                // steps 3/4) load global carry in from neighbor work-group
                //            and apply to local sub-group prefix carries
                if ((__sub_group_id == 0) && (__g > 0))
                {
                    auto __carry_offset = 0;

                    std::uint8_t __iters =
                        oneapi::dpl::__internal::__dpl_ceiling_div(__active_subgroups, __sub_group_size);

                    std::uint8_t __i = 0;
                    for (; __i < __iters - 1; ++__i)
                    {
                        __sub_group_partials[__carry_offset + __sub_group_local_id] =
                            __binary_op(__carry_last.__v, __sub_group_partials[__carry_offset + __sub_group_local_id]);
                        __carry_offset += __sub_group_size;
                    }
                    if (__i * __sub_group_size + __sub_group_local_id < __active_subgroups)
                    {
                        __sub_group_partials[__carry_offset + __sub_group_local_id] =
                            __binary_op(__carry_last.__v, __sub_group_partials[__carry_offset + __sub_group_local_id]);
                        __carry_offset += __sub_group_size;
                    }
                    if (__sub_group_local_id == 0)
                        __sub_group_partials[__active_subgroups] = __carry_last.__v;
                    __carry_last.__destroy();
                }
                __value.__destroy();

                //sycl::group_barrier(ndi.get_group());
                __ndi.barrier(sycl::access::fence_space::local_space);

                // Get inter-work group and adjusted for intra-work group prefix
                bool __sub_group_carry_initialized = true;
                if (__block_num == 0)
                {
                    if (__sub_group_id > 0)
                    {
                        auto __value = __sub_group_partials[__sub_group_id - 1];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __binary_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else if (__g > 0)
                    {
                        auto __value = __sub_group_partials[__active_subgroups];
                        oneapi::dpl::unseq_backend::__init_processing<_InitValueType>{}(__init, __value, __binary_op);
                        __sub_group_carry.__setup(__value);
                    }
                    else
                    {
                        if constexpr (std::is_same_v<_InitType, oneapi::dpl::unseq_backend::__no_init_value<_InitValueType>>)
                        {
                            // This is the only case where we still don't have a carry in.  No init value, 0th block,
                            // group, and subgroup. This changes the final scan through elements below.
                            __sub_group_carry_initialized = false;
                        }
                        else
                        {
                            __sub_group_carry.__setup(__init.__value);
                        }
                    }
                }
                else
                {
                    if (__sub_group_id > 0)
                    {
                        if constexpr (__is_inclusive)
                            __sub_group_carry.__setup(__binary_op(__out_rng[__block_num * __max_block_size - 1],
                                                                  __sub_group_partials[__sub_group_id - 1]));
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            _InValueType __last_block_element = __unary_op(__tmp_storage[__num_sub_groups_global]);
                            __sub_group_carry.__setup(__binary_op(
                                __binary_op(__out_rng[__block_num * __max_block_size - 1], __last_block_element),
                                __sub_group_partials[__sub_group_id - 1]));
                        }
                    }
                    else if (__g > 0)
                    {
                        if constexpr (__is_inclusive)
                            __sub_group_carry.__setup(__binary_op(__out_rng[__block_num * __max_block_size - 1],
                                                                  __sub_group_partials[__active_subgroups]));
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            _InValueType __last_block_element = __unary_op(__tmp_storage[__num_sub_groups_global]);
                            __sub_group_carry.__setup(__binary_op(
                                __binary_op(__out_rng[__block_num * __max_block_size - 1], __last_block_element),
                                __sub_group_partials[__active_subgroups]));
                        }
                    }
                    else
                    {
                        if constexpr (__is_inclusive)
                            __sub_group_carry.__setup(__out_rng[__block_num * __max_block_size - 1]);
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            _InValueType __last_block_element = __unary_op(__tmp_storage[__num_sub_groups_global]);
                            __sub_group_carry.__setup(
                                __binary_op(__out_rng[__block_num * __max_block_size - 1], __last_block_element));
                        }
                    }
                }
                // For the exclusive scan case:
                // Have the last item in the group store the last element
                // in the block to temporary storage for use in the next block.
                // This is required to support in-place exclusive scans as the input values will be overwritten.
                if constexpr (!__is_inclusive)
                {
                    auto __global_id = __ndi.get_global_linear_id();
                    if (__global_id == __num_work_items - 1)
                    {
                        std::size_t __last_idx_in_block = std::min(__n - 1, __max_block_size * (__block_num + 1) - 1);
                        __tmp_storage[__num_sub_groups_global] = __in_rng[__last_idx_in_block];
                    }
                }

                // step 5) apply global carries
                size_t __subgroup_start_idx = __group_start_idx + (__sub_group_id * __inputs_per_sub_group);
                size_t __start_idx = __subgroup_start_idx + __sub_group_local_id;

                if (__sub_group_carry_initialized)
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/true,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __unary_op, __binary_op, __sub_group_carry, __in_rng, __out_rng, __start_idx, __n,
                        __inputs_per_item, __subgroup_start_idx, __sub_group_id, __active_subgroups);

                    __sub_group_carry.__destroy();
                }
                else // first group first block, no subgroup carry
                {
                    __scan_through_elements_helper<__sub_group_size, __is_inclusive,
                                                   /*__init_present=*/false,
                                                   /*__capture_output=*/true, __max_inputs_per_item>(
                        __sub_group, __unary_op, __binary_op, __sub_group_carry, __in_rng, __out_rng, __start_idx, __n,
                        __inputs_per_item, __subgroup_start_idx, __sub_group_id, __active_subgroups);
                }
            });
        });
    }

    const sycl::nd_range<1> __nd_range;

    const std::size_t __max_block_size;
    const std::size_t __num_sub_groups_local;
    const std::size_t __num_sub_groups_global;
    const std::size_t __num_work_items;
    const std::size_t __n;

    const _BinaryOperation __binary_op;
    const _UnaryOperation __unary_op;
    _InitType __init;

    // TODO: Add the mask functors here to generalize for scan-based algorithms
};

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _BinaryOperation,
          typename _UnaryOperation, typename _InitType, typename _Inclusive>
auto
__parallel_transform_reduce_then_scan(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                      _InRng&& __in_rng, _OutRng&& __out_rng, _BinaryOperation __binary_op,
                                      _UnaryOperation __unary_op,
                                      _InitType __init /*TODO mask assigners for generalization go here*/, _Inclusive)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_reduce_kernel<_CustomName>>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __reduce_then_scan_scan_kernel<_CustomName>>;
    using _ValueType = typename _InitType::__value_type;

    constexpr std::size_t __sub_group_size = 32;
    // Empirically determined maximum. May be less for non-full blocks.
    constexpr std::uint8_t __max_inputs_per_item = 128;
    constexpr bool __inclusive = _Inclusive::value;

    // TODO: Do we need to adjust for slm usage or is the amount we use reasonably small enough
    // that no check is needed?
    const std::size_t __work_group_size = oneapi::dpl::__internal::__max_work_group_size(__exec);

    // TODO: base on max compute units. Recall disconnect in vendor definitions (# SMs vs. # XVEs)
    const std::size_t __num_work_groups = 128;
    const std::size_t __num_work_items = __num_work_groups * __work_group_size;
    const std::size_t __num_sub_groups_local = __work_group_size / __sub_group_size;
    const std::size_t __num_sub_groups_global = __num_sub_groups_local * __num_work_groups;
    const std::size_t __n = __in_rng.size();
    const std::size_t __max_inputs_per_block = __work_group_size * __max_inputs_per_item * __num_work_groups;
    std::size_t __num_remaining = __n;
    auto __inputs_per_sub_group =
        __n >= __max_inputs_per_block
            ? __max_inputs_per_block / __num_sub_groups_global
            : std::max(__sub_group_size,
                       oneapi::dpl::__internal::__dpl_bit_ceil(__num_remaining) / __num_sub_groups_global);
    auto __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
    const auto __global_range = sycl::range<1>(__num_work_items);
    const auto __local_range = sycl::range<1>(__work_group_size);
    const auto __kernel_nd_range = sycl::nd_range<1>(__global_range, __local_range);
    const auto __block_size = (__n < __max_inputs_per_block) ? __n : __max_inputs_per_block;
    const auto __num_blocks = __n / __block_size + (__n % __block_size != 0);

    // TODO: Use the trick in reduce to wrap in a shared_ptr with custom deleter to support asynchronous frees.
    _ValueType* __tmp_storage = sycl::malloc_device<_ValueType>(__num_sub_groups_global + 1, __exec.queue());

    // Reduce and scan step implementations
    using _ReduceSubmitter =
        __parallel_reduce_then_scan_reduce_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                     _BinaryOperation, _UnaryOperation, _InitType, _ReduceKernel>;
    using _ScanSubmitter =
        __parallel_reduce_then_scan_scan_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                   _BinaryOperation, _UnaryOperation, _InitType, _ScanKernel>;
    // TODO: remove below before merging. used for convenience now
    // clang-format off
    _ReduceSubmitter __reduce_submitter{__kernel_nd_range, __max_inputs_per_block, __num_sub_groups_local,
        __num_sub_groups_global, __num_work_items, __n, __binary_op, __unary_op, __init};
    _ScanSubmitter __scan_submitter{__kernel_nd_range, __max_inputs_per_block, __num_sub_groups_local, 
        __num_sub_groups_global, __num_work_items, __n, __binary_op, __unary_op, __init};
    // clang-format on

    sycl::event __event;
    // Data is processed in 2-kernel blocks to allow contiguous input segment to persist in LLC between the first and second kernel for accelerators
    // with sufficiently large L2 / L3 caches.
    for (std::size_t __b = 0; __b < __num_blocks; ++__b)
    {
        bool __is_full_block = __inputs_per_item == __max_inputs_per_item;
        // 1. Reduce step - Reduce assigned input per sub-group, compute and apply intra-wg carries, and write to global memory.
        __event = __reduce_submitter(__exec, __in_rng, __tmp_storage, __event, __inputs_per_sub_group,
                                     __inputs_per_item, __b, __is_full_block);
        // 2. Scan step - Compute intra-wg carries, determine sub-group carry-ins, and perform full input block scan.
        __event = __scan_submitter(__exec, __in_rng, __out_rng, __tmp_storage, __event, __inputs_per_sub_group,
                                   __inputs_per_item, __b, __is_full_block);
        if (__num_remaining > __block_size)
        {
            // Resize for the next block.
            __num_remaining -= __block_size;
            // TODO: This recalculation really only matters for the second to last iteration
            // of the loop since the last iteration is the only non-full block.
            __inputs_per_sub_group =
                __num_remaining >= __max_inputs_per_block
                    ? __max_inputs_per_block / __num_sub_groups_global
                    : std::max(__sub_group_size,
                               oneapi::dpl::__internal::__dpl_bit_ceil(__num_remaining) / __num_sub_groups_global);
            __inputs_per_item = __inputs_per_sub_group / __sub_group_size;
        }
    }
    // TODO: Remove to make asynchronous. Depends on completing async USM free TODO.
    __event.wait();
    sycl::free(__tmp_storage, __exec.queue());
    return __future(__event);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_THEN_SCAN_H
