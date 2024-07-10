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

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <iterator>
#include <cmath>
#include <tuple>

#include "internal/esimd_defs.h"
#include "../../pstl/utils.h"
#include "../../pstl/hetero/dpcpp/unseq_backend_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

namespace oneapi::dpl::experimental::kt::gpu
{

namespace __impl
{
template <typename... _Name>
class __two_pass_scan_kernel1;

template <typename... _Name>
class __two_pass_scan_kernel2;

template <std::uint8_t VL, bool InitPresent, typename MaskOp, typename InitBroadcastId, typename SubGroup,
          typename BinaryOp, typename ValueType, typename LazyValueType>
void
exclusive_sub_group_masked_scan(const SubGroup& sub_group, MaskOp mask_fn, InitBroadcastId init_broadcast_id,
                                ValueType& value, BinaryOp binary_op, LazyValueType& init_and_carry)
{

    std::uint8_t sub_group_local_id = sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t shift = 1; shift <= VL / 2; shift <<= 1)
    {
        auto partial_carry_in = sycl::shift_group_right(sub_group, value, shift);
        if (mask_fn(sub_group_local_id, shift))
        {
            value = binary_op(partial_carry_in, value);
        }
    }
    LazyValueType old_init;
    if constexpr (InitPresent)
    {
        value = binary_op(init_and_carry.__v, value);
        if (sub_group_local_id == 0)
            old_init.__setup(init_and_carry.__v);
        init_and_carry.__v = sycl::group_broadcast(sub_group, value, init_broadcast_id);
    }
    else
    {
        init_and_carry.__setup(sycl::group_broadcast(sub_group, value, init_broadcast_id));
    }

    value = sycl::shift_group_right(sub_group, value, 1);
    if constexpr (InitPresent)
    {
        if (sub_group_local_id == 0)
        {
            value = old_init.__v;
            old_init.__destroy();
        }
    }
    //return by reference value and init_and_carry
}

template <std::uint8_t VL, bool InitPresent, typename MaskOp, typename InitBroadcastId, typename SubGroup,
          typename BinaryOp, typename ValueType, typename LazyValueType>
void
inclusive_sub_group_masked_scan(const SubGroup& sub_group, MaskOp mask_fn, InitBroadcastId init_broadcast_id,
                                ValueType& value, BinaryOp binary_op, LazyValueType& init_and_carry)
{

    std::uint8_t sub_group_local_id = sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t shift = 1; shift <= VL / 2; shift <<= 1)
    {
        auto partial_carry_in = sycl::shift_group_right(sub_group, value, shift);
        if (mask_fn(sub_group_local_id, shift))
        {
            value = binary_op(partial_carry_in, value);
        }
    }
    if constexpr (InitPresent)
    {
        value = binary_op(init_and_carry.__v, value);
        init_and_carry.__v = sycl::group_broadcast(sub_group, value, init_broadcast_id);
    }
    else
    {
        init_and_carry.__setup(sycl::group_broadcast(sub_group, value, init_broadcast_id));
    }

    //return by reference value and init_and_carry
}

template <std::uint8_t VL, bool IsInclusive, bool InitPresent, typename SubGroup, typename BinaryOp, typename ValueType,
          typename LazyValueType>
void
sub_group_scan(const SubGroup& sub_group, ValueType& value, BinaryOp binary_op, LazyValueType& init_and_carry)
{
    auto mask_fn = [](auto sub_group_local_id, auto offset) { return sub_group_local_id >= offset; };
    constexpr auto init_broadcast_id = VL - 1;
    if constexpr (IsInclusive)
    {
        inclusive_sub_group_masked_scan<VL, InitPresent>(sub_group, mask_fn, init_broadcast_id, value, binary_op,
                                                         init_and_carry);
    }
    else
    {
        exclusive_sub_group_masked_scan<VL, InitPresent>(sub_group, mask_fn, init_broadcast_id, value, binary_op,
                                                         init_and_carry);
    }
}

template <std::uint8_t VL, bool IsInclusive, bool InitPresent, typename SubGroup, typename BinaryOp, typename ValueType,
          typename LazyValueType, typename SizeType>
void
sub_group_scan_partial(const SubGroup& sub_group, ValueType& value, BinaryOp binary_op, LazyValueType& init_and_carry,
                       SizeType elements_to_process)
{
    auto mask_fn = [elements_to_process](auto sub_group_local_id, auto offset) {
        return sub_group_local_id >= offset && sub_group_local_id < elements_to_process;
    };
    auto init_broadcast_id = elements_to_process - 1;
    if constexpr (IsInclusive)
    {
        inclusive_sub_group_masked_scan<VL, InitPresent>(sub_group, mask_fn, init_broadcast_id, value, binary_op,
                                                         init_and_carry);
    }
    else
    {
        exclusive_sub_group_masked_scan<VL, InitPresent>(sub_group, mask_fn, init_broadcast_id, value, binary_op,
                                                         init_and_carry);
    }
}

template <std::uint8_t VL, bool IsInclusive, bool InitPresent, bool CaptureOutput, std::uint32_t J_max,
          typename SubGroup, typename UnaryOp, typename BinaryOp, typename LazyValueType, typename InRange,
          typename OutRange>
void
scan_through_elements_helper(const SubGroup& sub_group, UnaryOp unary_op, BinaryOp binary_op,
                             LazyValueType& sub_group_carry, InRange __in_rng, OutRange __out_rng,
                             std::size_t start_idx, std::size_t M, std::uint32_t J, std::size_t subgroup_start_idx,
                             std::uint32_t sub_group_id, std::uint32_t active_subgroups)
{
    bool is_full_block = (J == J_max);
    bool is_full_thread = subgroup_start_idx + J * VL <= M;
    if (is_full_thread && is_full_block)
    {
        auto v = unary_op(__in_rng[start_idx]);
        sub_group_scan<VL, IsInclusive, InitPresent>(sub_group, v, binary_op, sub_group_carry);
        if constexpr (CaptureOutput)
        {
            __out_rng[start_idx] = v;
        }

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t j = 1; j < J_max; j++)
        {
            v = unary_op(__in_rng[start_idx + j * VL]);
            sub_group_scan<VL, IsInclusive, /*InitPresent=*/true>(sub_group, v, binary_op, sub_group_carry);
            if constexpr (CaptureOutput)
            {
                __out_rng[start_idx + j * VL] = v;
            }
        }
    }
    else if (is_full_thread)
    {
        auto v = unary_op(__in_rng[start_idx]);
        sub_group_scan<VL, IsInclusive, InitPresent>(sub_group, v, binary_op, sub_group_carry);
        if constexpr (CaptureOutput)
        {
            __out_rng[start_idx] = v;
        }
        for (std::uint32_t j = 1; j < J; j++)
        {
            v = unary_op(__in_rng[start_idx + j * VL]);
            sub_group_scan<VL, IsInclusive, /*InitPresent=*/true>(sub_group, v, binary_op, sub_group_carry);
            if constexpr (CaptureOutput)
            {
                __out_rng[start_idx + j * VL] = v;
            }
        }
    }
    else
    {
        if (sub_group_id < active_subgroups)
        {
            auto iters = oneapi::dpl::__internal::__dpl_ceiling_div(M - subgroup_start_idx, VL);

            if (iters == 1)
            {
                auto v = unary_op(__in_rng[start_idx]);
                sub_group_scan_partial<VL, IsInclusive, InitPresent>(sub_group, v, binary_op, sub_group_carry,
                                                                     M - subgroup_start_idx);
                if constexpr (CaptureOutput)
                {
                    if (start_idx < M)
                        __out_rng[start_idx] = v;
                }
            }
            else
            {
                auto v = unary_op(__in_rng[start_idx]);
                sub_group_scan<VL, IsInclusive, InitPresent>(sub_group, v, binary_op, sub_group_carry);
                if constexpr (CaptureOutput)
                {
                    __out_rng[start_idx] = v;
                }

                for (int j = 1; j < iters - 1; j++)
                {
                    auto local_idx = start_idx + j * VL;
                    v = unary_op(__in_rng[local_idx]);
                    sub_group_scan<VL, IsInclusive, /*InitPresent=*/true>(sub_group, v, binary_op, sub_group_carry);
                    if constexpr (CaptureOutput)
                    {
                        __out_rng[local_idx] = v;
                    }
                }

                auto offset = start_idx + (iters - 1) * VL;
                auto local_idx = (offset < M) ? offset : M - 1;
                v = unary_op(__in_rng[local_idx]);
                sub_group_scan_partial<VL, IsInclusive, /*InitPresent=*/true>(
                    sub_group, v, binary_op, sub_group_carry, M - (subgroup_start_idx + (iters - 1) * VL));
                if constexpr (CaptureOutput)
                {
                    if (offset < M)
                        __out_rng[offset] = v;
                }
            }
        }
    }
}

// Named two_pass_scan for now to avoid name clash with single pass KT
template <bool IsInclusive, typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp,
          typename UnaryOp, typename InitType>
void
two_pass_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng, BinaryOp binary_op, UnaryOp unary_op,
              InitType init)
{
    using namespace sycl;
    using InValueType = oneapi::dpl::__internal::__value_t<_InRng>;
    // PVC 1 tile
    constexpr std::uint32_t log2_VL = 5;
    constexpr std::uint32_t VL = 1 << log2_VL; // simd vector length 2^5 = 32

    std::uint32_t work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    // TODO: develop simple heuristic to determine this value based on maximum compute units
    std::uint32_t num_work_groups = 128;
    std::uint32_t num_work_items = work_group_size * num_work_groups;
    std::uint32_t num_sub_groups_local = work_group_size / VL;
    std::uint32_t num_sub_groups_global = num_sub_groups_local * num_work_groups;

    using _FirstKernel = /*TODO: oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<*/
        __two_pass_scan_kernel1<_KernelName, InValueType, BinaryOp>;

    using _SecondKernel = /*TODO: oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<*/
        __two_pass_scan_kernel2<_KernelName, InValueType, BinaryOp>;

    size_t M = __in_rng.size();
    size_t num_remaining = M;

    constexpr std::uint32_t J_max = 128;
    std::size_t MAX_INPUTS_PER_BLOCK =
        work_group_size * J_max * num_work_groups; // empirically determined for reduce_then_scan
    // items per PVC hardware thread
    std::uint32_t K =
        (M >= MAX_INPUTS_PER_BLOCK)
            ? MAX_INPUTS_PER_BLOCK / num_sub_groups_global
            : std::max(std::size_t(VL), oneapi::dpl::__internal::__dpl_bit_ceil(num_remaining) / num_sub_groups_global);
    // SIMD vectors per PVC hardware thread
    std::uint32_t J = K / VL;

    auto blockSize = (M < MAX_INPUTS_PER_BLOCK) ? M : MAX_INPUTS_PER_BLOCK;
    auto numBlocks = M / blockSize + (M % blockSize != 0);

    auto globalRange = range<1>(num_work_items);
    auto localRange = range<1>(work_group_size);
    nd_range<1> range(globalRange, localRange);

    // Each element is a partial result from a subgroup. The last element is to support in-place
    // exclusive scans where we need to store the original input's last element for future use
    // before it is overwritten.
    InValueType* tmp_storage = sycl::malloc_device<InValueType>(num_sub_groups_global + 1, q);
    sycl::event event;

    // run scan kernels for all input blocks in the current buffer
    // e.g., scan length 2^24 / MAX_INPUTS_PER_BLOCK = 2 blocks
    for (std::size_t b = 0; b < numBlocks; b++)
    {
        // the first kernel computes sub-group local prefix scans and
        // subgroup local carries, one per thread
        // intermediate partial sums and carries write back to the output buffer
        event = q.submit([&](handler& h) {
            sycl::local_accessor<InValueType> sub_group_partials(num_sub_groups_local, h);
            h.depends_on(event);
            oneapi::dpl::__ranges::__require_access(h, __in_rng);
            h.parallel_for<_FirstKernel>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
                auto id = ndi.get_global_id(0);
                auto lid = ndi.get_local_id(0);
                auto g = ndi.get_group(0);
                auto sub_group = ndi.get_sub_group();
                auto sub_group_id = sub_group.get_group_linear_id();
                auto sub_group_local_id = sub_group.get_local_linear_id();

                oneapi::dpl::__internal::__lazy_ctor_storage<InValueType> sub_group_carry;
                std::size_t group_start_idx = (b * blockSize) + (g * K * num_sub_groups_local);
                if (M <= group_start_idx)
                    return; // exit early for empty groups (TODO: avoid launching these?)

                std::size_t elements_in_group = std::min(M - group_start_idx, std::size_t(num_sub_groups_local * K));
                std::uint32_t active_subgroups = oneapi::dpl::__internal::__dpl_ceiling_div(elements_in_group, K);
                std::size_t subgroup_start_idx = group_start_idx + (sub_group_id * K);

                std::size_t start_idx = subgroup_start_idx + sub_group_local_id;

                if (sub_group_id < active_subgroups)
                {
                    // adjust for lane-id
                    // compute sub-group local pfix on T0..63, K samples/T, send to accumulator kernel
                    scan_through_elements_helper<VL, IsInclusive,
                                                 /*InitPresent=*/false,
                                                 /*CaptureOutput=*/false, J_max>(
                        sub_group, unary_op, binary_op, sub_group_carry, __in_rng, nullptr, start_idx, M, J,
                        subgroup_start_idx, sub_group_id, active_subgroups);
                    if (sub_group_local_id == 0)
                        sub_group_partials[sub_group_id] = sub_group_carry.__v;
                    sub_group_carry.__destroy();
                }
                // TODO: This is slower then ndi.barrier which was removed in SYCL2020. Can we do anything about it?
                //sycl::group_barrier(ndi.get_group());
                ndi.barrier(sycl::access::fence_space::local_space);

                // compute sub-group local prefix sums on (T0..63) carries
                // and store to scratch space at the end of dst; next
                // accumulator kernel takes M thread carries from scratch
                // to compute a prefix sum on global carries
                if (sub_group_id == 0)
                {
                    start_idx = (g * num_sub_groups_local);
                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(active_subgroups, VL);
                    if (iters == 1)
                    {
                        auto load_idx = (sub_group_local_id < active_subgroups)
                                            ? sub_group_local_id
                                            : (active_subgroups - 1); // else is unused dummy value
                        auto v = sub_group_partials[load_idx];
                        sub_group_scan_partial<VL, /*IsInclusive=*/true, /*InitPresent=*/false>(
                            sub_group, v, binary_op, sub_group_carry, active_subgroups - subgroup_start_idx);
                        if (sub_group_local_id < active_subgroups)
                            tmp_storage[start_idx + sub_group_local_id] = v;
                    }
                    else
                    {
                        //need to pull out first iteration tp avoid identity
                        auto v = sub_group_partials[sub_group_local_id];
                        sub_group_scan<VL, /*IsInclusive=*/true, /*InitPresent=*/false>(sub_group, v, binary_op,
                                                                                        sub_group_carry);
                        tmp_storage[start_idx + sub_group_local_id] = v;

                        for (int i = 1; i < iters - 1; i++)
                        {
                            v = sub_group_partials[i * VL + sub_group_local_id];
                            sub_group_scan<VL, /*IsInclusive=*/true, /*InitPresent=*/true>(sub_group, v, binary_op,
                                                                                           sub_group_carry);
                            tmp_storage[start_idx + i * VL + sub_group_local_id] = v;
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.

                        // else is an unused dummy value
                        auto proposed_idx = (iters - 1) * VL + sub_group_local_id;
                        auto load_idx =
                            (proposed_idx < num_sub_groups_local) ? proposed_idx : (num_sub_groups_local - 1);

                        v = sub_group_partials[load_idx];
                        sub_group_scan_partial<VL, /*IsInclusive=*/true, /*InitPresent=*/true>(
                            sub_group, v, binary_op, sub_group_carry, num_sub_groups_local);
                        if (proposed_idx < num_sub_groups_local)
                            tmp_storage[start_idx + proposed_idx] = v;
                    }

                    sub_group_carry.__destroy();
                }
            });
        });

        // the second kernel computes a prefix sum on sub-group local carries
        // then propogates carries inter-WG to generate thread-local versions
        // of the global carries on each sub-group; then the ouput stage
        // recomputes the thread-local prefix scan - add carry - store to compute the final sum
        event = q.submit([&](handler& CGH) {
            sycl::local_accessor<InValueType> sub_group_partials(num_sub_groups_local + 1, CGH);
            CGH.depends_on(event);
            oneapi::dpl::__ranges::__require_access(CGH, __in_rng, __out_rng);
            CGH.parallel_for<_SecondKernel>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
                auto id = ndi.get_global_id(0);
                auto lid = ndi.get_local_id(0);
                auto g = ndi.get_group(0);
                auto sub_group = ndi.get_sub_group();
                auto sub_group_id = sub_group.get_group_linear_id();
                auto sub_group_local_id = sub_group.get_local_linear_id();

                auto group_start_idx = (b * blockSize) + (g * K * num_sub_groups_local);
                if (M <= group_start_idx)
                    return; // exit early for empty groups (TODO: avoid launching these?)

                std::size_t elements_in_group = std::min(M - group_start_idx, std::size_t(num_sub_groups_local * K));
                std::uint32_t active_subgroups = oneapi::dpl::__internal::__dpl_ceiling_div(elements_in_group, K);
                oneapi::dpl::__internal::__lazy_ctor_storage<InValueType> carry_last;
                oneapi::dpl::__internal::__lazy_ctor_storage<InValueType> value;

                // propogate carry in from previous block
                oneapi::dpl::__internal::__lazy_ctor_storage<InValueType> sub_group_carry;

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
                if (sub_group_id == 0)
                {
                    // step 1) load to Xe slm the WG-local S prefix sums
                    //         on WG T-local carries
                    //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
                    //           S: sum(T0 carry...TS carry)
                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(active_subgroups, VL);
                    auto subgroups_before_my_group = g * num_sub_groups_local;
                    std::uint8_t i = 0;
                    for (; i < iters - 1; i++)
                    {
                        sub_group_partials[i * VL + sub_group_local_id] =
                            tmp_storage[subgroups_before_my_group + i * VL + sub_group_local_id];
                    }
                    if (i * VL + sub_group_local_id < active_subgroups)
                    {
                        sub_group_partials[i * VL + sub_group_local_id] =
                            tmp_storage[subgroups_before_my_group + i * VL + sub_group_local_id];
                    }

                    // step 2) load 32, 64, 96, etc. work-group carry outs on every work-group; then
                    //         compute the prefix in a sub-group to get global work-group carries
                    //         memory accesses: gather(63, 127, 191, 255, ...)
                    uint32_t offset = num_sub_groups_local - 1;
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    if (g > 0)
                    {
                        // only need the last element from each scan of num_sub_groups_local subgroup reductions
                        const auto elements_to_process = subgroups_before_my_group / num_sub_groups_local;
                        const auto pre_carry_iters =
                            oneapi::dpl::__internal::__dpl_ceiling_div(elements_to_process, VL);
                        if (pre_carry_iters == 1)
                        {
                            // single partial scan
                            auto proposed_idx = num_sub_groups_local * sub_group_local_id + offset;
                            auto remaining_elements = elements_to_process;
                            auto reduction_idx = (proposed_idx < subgroups_before_my_group)
                                                     ? proposed_idx
                                                     : subgroups_before_my_group - 1;
                            value.__setup(tmp_storage[reduction_idx]);
                            sub_group_scan_partial<VL, /*IsInclusive=*/true, /*InitPresent=*/false>(
                                sub_group, value.__v, binary_op, carry_last, remaining_elements);
                        }
                        else
                        {
                            // multiple iterations
                            // first 1 full
                            value.__setup(tmp_storage[num_sub_groups_local * sub_group_local_id + offset]);
                            sub_group_scan<VL, /*IsInclusive=*/true, /*InitPresent=*/false>(sub_group, value.__v,
                                                                                            binary_op, carry_last);

                            // then some number of full iterations
                            for (int i = 1; i < pre_carry_iters - 1; i++)
                            {
                                auto reduction_idx =
                                    i * num_sub_groups_local * VL + num_sub_groups_local * sub_group_local_id + offset;
                                value.__v = tmp_storage[reduction_idx];
                                sub_group_scan<VL, /*IsInclusive=*/true, /*InitPresent=*/true>(sub_group, value.__v,
                                                                                               binary_op, carry_last);
                            }

                            // final partial iteration
                            auto proposed_idx = (pre_carry_iters - 1) * num_sub_groups_local * VL +
                                                num_sub_groups_local * sub_group_local_id + offset;
                            auto remaining_elements = elements_to_process - ((pre_carry_iters - 1) * VL);
                            auto reduction_idx = (proposed_idx < subgroups_before_my_group)
                                                     ? proposed_idx
                                                     : subgroups_before_my_group - 1;
                            value.__v = tmp_storage[reduction_idx];
                            sub_group_scan_partial<VL, /*IsInclusive=*/true, /*InitPresent=*/true>(
                                sub_group, value.__v, binary_op, carry_last, remaining_elements);
                        }
                    }
                }
                // For the exclusive scan case:
                // While the first sub-group is doing work, have the last item in the group store the last element
                // in the block to temporary storage for use in the next block.
                // This is required to support in-place exclusive scans as the input values will be overwritten.
                if constexpr (!IsInclusive)
                {
                    auto global_id = ndi.get_global_linear_id();
                    if (global_id == num_work_items - 1)
                    {
                        std::size_t last_idx_in_block = std::min(M - 1, blockSize * (b + 1) - 1);
                        tmp_storage[num_sub_groups_global] = __in_rng[last_idx_in_block];
                    }
                }

                // N.B. barrier could be earlier, guarantees slm local carry update
                //sycl::group_barrier(ndi.get_group());
                ndi.barrier(sycl::access::fence_space::local_space);

                // steps 3/4) load global carry in from neighbor work-group
                //            and apply to local sub-group prefix carries
                if ((sub_group_id == 0) && (g > 0))
                {
                    auto carry_offset = 0;

                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(active_subgroups, VL);

                    std::uint8_t i = 0;
                    for (; i < iters - 1; ++i)
                    {
                        sub_group_partials[carry_offset + sub_group_local_id] =
                            binary_op(carry_last.__v, sub_group_partials[carry_offset + sub_group_local_id]);
                        carry_offset += VL;
                    }
                    if (i * VL + sub_group_local_id < active_subgroups)
                    {
                        sub_group_partials[carry_offset + sub_group_local_id] =
                            binary_op(carry_last.__v, sub_group_partials[carry_offset + sub_group_local_id]);
                        carry_offset += VL;
                    }
                    if (sub_group_local_id == 0)
                        sub_group_partials[active_subgroups] = carry_last.__v;
                    carry_last.__destroy();
                }
                value.__destroy();

                //sycl::group_barrier(ndi.get_group());
                ndi.barrier(sycl::access::fence_space::local_space);

                // Get inter-work group and adjusted for intra-work group prefix
                bool sub_group_carry_initialized = true;
                if (b == 0)
                {
                    if (sub_group_id > 0)
                    {
                        auto value = sub_group_partials[sub_group_id - 1];
                        oneapi::dpl::unseq_backend::__init_processing<InValueType>{}(init, value, binary_op);
                        sub_group_carry.__setup(value);
                    }
                    else if (g > 0)
                    {
                        auto value = sub_group_partials[active_subgroups];
                        oneapi::dpl::unseq_backend::__init_processing<InValueType>{}(init, value, binary_op);
                        sub_group_carry.__setup(value);
                    }
                    else
                    {
                        if constexpr (std::is_same_v<InitType, oneapi::dpl::unseq_backend::__no_init_value<
                                                                   typename InitType::__value_type>>)
                        {
                            // This is the only case where we still don't have a carry in.  No init value, 0th block,
                            // group, and subgroup. This changes the final scan through elements below.
                            sub_group_carry_initialized = false;
                        }
                        else
                        {
                            sub_group_carry.__setup(init.__value);
                        }
                    }
                }
                else
                {
                    if (sub_group_id > 0)
                    {
                        if constexpr (IsInclusive)
                            sub_group_carry.__setup(
                                binary_op(__out_rng[b * blockSize - 1], sub_group_partials[sub_group_id - 1]));
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            InValueType last_block_element = unary_op(tmp_storage[num_sub_groups_global]);
                            sub_group_carry.__setup(
                                binary_op(binary_op(__out_rng[b * blockSize - 1], last_block_element),
                                          sub_group_partials[sub_group_id - 1]));
                        }
                    }
                    else if (g > 0)
                    {
                        if constexpr (IsInclusive)
                            sub_group_carry.__setup(
                                binary_op(__out_rng[b * blockSize - 1], sub_group_partials[active_subgroups]));
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            InValueType last_block_element = unary_op(tmp_storage[num_sub_groups_global]);
                            sub_group_carry.__setup(
                                binary_op(binary_op(__out_rng[b * blockSize - 1], last_block_element),
                                          sub_group_partials[active_subgroups]));
                        }
                    }
                    else
                    {
                        if constexpr (IsInclusive)
                            sub_group_carry.__setup(__out_rng[b * blockSize - 1]);
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the second kernel of the previous block.
                            InValueType last_block_element = unary_op(tmp_storage[num_sub_groups_global]);
                            sub_group_carry.__setup(binary_op(__out_rng[b * blockSize - 1], last_block_element));
                        }
                    }
                }

                // step 5) apply global carries
                size_t subgroup_start_idx = group_start_idx + (sub_group_id * K);
                size_t start_idx = subgroup_start_idx + sub_group_local_id;

                if (sub_group_carry_initialized)
                {
                    scan_through_elements_helper<VL, IsInclusive,
                                                 /*InitPresent=*/true,
                                                 /*CaptureOutput=*/true, J_max>(
                        sub_group, unary_op, binary_op, sub_group_carry, __in_rng, __out_rng, start_idx, M, J,
                        subgroup_start_idx, sub_group_id, active_subgroups);

                    sub_group_carry.__destroy();
                }
                else // first group first block, no subgroup carry
                {
                    scan_through_elements_helper<VL, IsInclusive,
                                                 /*InitPresent=*/false,
                                                 /*CaptureOutput=*/true, J_max>(
                        sub_group, unary_op, binary_op, sub_group_carry, __in_rng, __out_rng, start_idx, M, J,
                        subgroup_start_idx, sub_group_id, active_subgroups);
                }
            });
        });
        // Resize K and J for the last block
        if (num_remaining > blockSize)
        {
            num_remaining -= blockSize;
            // TODO: add support to invoke a single work-group implementation on either the last iteration
            K = num_remaining >= MAX_INPUTS_PER_BLOCK
                    ? MAX_INPUTS_PER_BLOCK / num_sub_groups_global
                    : std::max(std::size_t(VL),
                               oneapi::dpl::__internal::__dpl_bit_ceil(num_remaining) / num_sub_groups_global);
            // SIMD vectors per PVC hardware thread
            J = K / VL;
        }
    } // block
    event.wait();
    sycl::free(tmp_storage, q);
}

} // namespace __impl

namespace ranges
{

template <typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp, typename UnaryOp,
          typename ValueType>
void
two_pass_transform_exclusive_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng, BinaryOp binary_op,
                                  UnaryOp unary_op, ValueType init)
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    __impl::two_pass_scan<false, _KernelName>(q, std::move(__in_view), std::move(__out_view), binary_op, unary_op,
                                              oneapi::dpl::unseq_backend::__init_value<ValueType>{init});
}

template <typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp, typename UnaryOp,
          typename ValueType>
void
two_pass_transform_inclusive_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng, BinaryOp binary_op,
                                  UnaryOp unary_op, ValueType init)
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    __impl::two_pass_scan<true, _KernelName>(q, std::move(__in_view), std::move(__out_view), binary_op, unary_op,
                                             oneapi::dpl::unseq_backend::__init_value<ValueType>{init});
}

template <typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp, typename UnaryOp>
void
two_pass_transform_inclusive_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng, BinaryOp binary_op,
                                  UnaryOp unary_op)
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    __impl::two_pass_scan<true, _KernelName>(q, std::move(__in_view), std::move(__out_view), binary_op, unary_op,
                                             oneapi::dpl::unseq_backend::__no_init_value{});
}

} // namespace ranges

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename BinaryOp, typename UnaryOp, typename ValueType>
void
two_pass_transform_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                                  BinaryOp binary_op, UnaryOp unary_op, ValueType init)
{
    auto __n = last - first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, InputIterator>();
    auto __buf1 = __keep1(first, last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, OutputIterator>();
    auto __buf2 = __keep2(result, result + __n);

    ranges::two_pass_transform_inclusive_scan<_KernelName>(q, __buf1.all_view(), __buf2.all_view(), binary_op, unary_op,
                                                           init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename BinaryOp, typename UnaryOp>
void
two_pass_transform_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                                  BinaryOp binary_op, UnaryOp unary_op)
{
    auto __n = last - first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, InputIterator>();
    auto __buf1 = __keep1(first, last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, OutputIterator>();
    auto __buf2 = __keep2(result, result + __n);

    ranges::two_pass_transform_inclusive_scan<_KernelName>(q, __buf1.all_view(), __buf2.all_view(), binary_op,
                                                           unary_op);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename ValueType, typename BinaryOp, typename UnaryOp>
void
two_pass_transform_exclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                                  ValueType init, BinaryOp binary_op, UnaryOp unary_op)
{
    auto __n = last - first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, InputIterator>();
    auto __buf1 = __keep1(first, last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, OutputIterator>();
    auto __buf2 = __keep2(result, result + __n);

    ranges::two_pass_transform_exclusive_scan<_KernelName>(q, __buf1.all_view(), __buf2.all_view(), binary_op, unary_op,
                                                           init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename BinaryOp, typename ValueType>
void
two_pass_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        BinaryOp binary_op, ValueType init)
{
    two_pass_transform_inclusive_scan<_KernelName>(q, first, last, result, binary_op,
                                                   oneapi::dpl::__internal::__no_op(), init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename BinaryOp>
void
two_pass_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        BinaryOp binary_op)
{
    two_pass_transform_inclusive_scan<_KernelName>(q, first, last, result, binary_op,
                                                   oneapi::dpl::__internal::__no_op());
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator,
          typename OutputIterator, typename BinaryOp, typename ValueType>
void
two_pass_exclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result, ValueType init,
                        BinaryOp binary_op)
{
    two_pass_transform_exclusive_scan<_KernelName>(q, first, last, result, init, binary_op,
                                                   oneapi::dpl::__internal::__no_op());
}

} // namespace oneapi::dpl::experimental::kt::gpu
