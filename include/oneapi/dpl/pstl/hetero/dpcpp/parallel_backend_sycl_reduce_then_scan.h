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

// TODO: Scan related specific utilities will be placed here

template <typename... _Name>
class __reduce_then_scan_reduce_kernel;

template <typename... _Name>
class __reduce_then_scan_scan_kernel;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __inclusive, typename _BinaryOperation,
          typename _UnaryOperation, typename _WrappedInitType, typename _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __inclusive, typename _BinaryOperation,
          typename _UnaryOperation, typename _WrappedInitType, typename... _KernelName>
struct __parallel_reduce_then_scan_reduce_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                    _BinaryOperation, _UnaryOperation, _WrappedInitType,
                                                    __internal::__optional_kernel_name<_KernelName...>>
{
    // Step 1 - SubGroupReduce is expected to perform sub-group reductions to global memory
    // input buffer
    template <typename _ExecutionPolicy, typename _Range, typename _TmpStorageAcc>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, _TmpStorageAcc __tmp_storage_acc,
               const sycl::event& __prior_event, const std::size_t __inputs_per_sub_group,
               const std::size_t __inputs_per_item, const std::size_t __block_num, const bool __is_full_block) const
    {
        return __exec.queue().submit([&, this](sycl::handler& __cgh) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __cgh.parallel_for<_KernelName...>(__nd_range, [=, *this](sycl::nd_item<1> __ndi) {
                // TODO: Add kernel body
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
    const _WrappedInitType __wrapped_init;

    // TODO: Add the mask functors here to generalize for scan-based algorithms
};

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __inclusive, typename _BinaryOperation,
          typename _UnaryOperation, typename _WrappedInitType, typename _KernelName>
struct __parallel_reduce_then_scan_scan_submitter;

template <std::size_t __sub_group_size, std::size_t __max_inputs_per_item, bool __inclusive, typename _BinaryOperation,
          typename _UnaryOperation, typename _WrappedInitType, typename... _KernelName>
struct __parallel_reduce_then_scan_scan_submitter<__sub_group_size, __max_inputs_per_item, __inclusive,
                                                  _BinaryOperation, _UnaryOperation, _WrappedInitType,
                                                  __internal::__optional_kernel_name<_KernelName...>>
{
    template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _TmpStorageAcc>
    auto
    operator()(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _TmpStorageAcc __tmp_storage_acc,
               const sycl::event& __prior_event, const std::size_t __inputs_per_sub_group,
               const std::size_t __inputs_per_item, const std::size_t __block_num, const bool __is_full_block) const
    {
        return __exec.queue().submit([&, this](sycl::handler& __cgh) [[sycl::reqd_sub_group_size(__sub_group_size)]] {
            __cgh.depends_on(__prior_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2);
            __cgh.parallel_for<_KernelName...>(__nd_range, [=, *this](sycl::nd_item<1> __ndi) {
                // TODO: Add kernel body
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
    const _WrappedInitType __wrapped_init;

    // TODO: Add the mask functors here to generalize for scan-based algorithms
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryOperation,
          typename _UnaryOperation, typename _InitType, typename _Inclusive>
auto
__parallel_transform_reduce_then_scan(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                      _Range1&& __in_rng, _Range2&& __out_rng, _BinaryOperation __binary_op,
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