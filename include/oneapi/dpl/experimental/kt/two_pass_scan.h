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


template <std::uint8_t VL, bool Inclusive, typename MaskOp, typename InitBroadcastId, typename SubGroup, typename BinaryOp, typename ValueType>
std::tuple<ValueType, ValueType>
sub_group_masked_scan(const SubGroup& sub_group, MaskOp mask_fn, InitBroadcastId init_broadcast_id,
                      ValueType value, BinaryOp binary_op, ValueType init)
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
    value = binary_op(init, value);
    // Adjust for an exclusive scan if requested
    if constexpr (!Inclusive)
    {
        auto old_init = init;
        init = sycl::group_broadcast(sub_group, value, init_broadcast_id);
        value = sycl::shift_group_right(sub_group, value, 1);
        if (sub_group_local_id == 0)
            value = old_init;
    }
    else
    {
        init = sycl::group_broadcast(sub_group, value, init_broadcast_id);
    }
    return std::make_tuple(std::move(value), std::move(init));
}

template <std::uint8_t VL, bool Inclusive, typename SubGroup, typename BinaryOp, typename ValueType>
std::tuple<ValueType, ValueType>
sub_group_scan(const SubGroup& sub_group, ValueType value, BinaryOp binary_op, ValueType init)
{
    auto mask_fn = [](auto sub_group_local_id, auto offset) { return sub_group_local_id >= offset; };
    constexpr auto init_broadcast_id = VL - 1;
    return sub_group_masked_scan<VL, Inclusive>(sub_group, mask_fn, init_broadcast_id, value, binary_op, init);
}

template <std::uint8_t VL, bool Inclusive, typename SubGroup, typename BinaryOp, typename ValueType, typename SizeType>
std::tuple<ValueType, ValueType>
sub_group_scan(const SubGroup& sub_group, ValueType value, BinaryOp binary_op, ValueType init, SizeType num_remaining)
{
    auto mask_fn = [num_remaining](auto sub_group_local_id, auto offset) {
        return sub_group_local_id >= offset && sub_group_local_id < num_remaining;
    };
    auto init_broadcast_id = num_remaining - 1;
    return sub_group_masked_scan<VL, Inclusive>(sub_group, mask_fn, init_broadcast_id, value, binary_op, init);
}

// Named two_pass_scan for now to avoid name clash with single pass KT
template <bool Inclusive, typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp, typename UnaryOp, typename ValueType>
void
two_pass_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng,
              BinaryOp binary_op, UnaryOp unary_op, ValueType init)
{
    using namespace sycl;

    // PVC 1 tile
    constexpr std::uint32_t log2_VL = 5;
    constexpr std::uint32_t VL = 1 << log2_VL;               // simd vector length 2^5 = 32

    std::uint32_t work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    // TODO: develop simple heuristic to determine this value based on maximum compute units
    std::uint32_t num_work_groups = 128;
    std::uint32_t num_sub_groups_local = work_group_size / VL;
    std::uint32_t num_sub_groups_global = num_sub_groups_local * num_work_groups;
    // Is set if the scanner sub-group that scans sub-group carries in a work-group divides evenly into
    // the vector length (e.g. With a VL of 32 and work-group size of 1024, we have 32 sub-groups which makes this true).
    // The number of sub-groups being a multiple of the vector length is preferred for performance. 
    bool is_full_carry_scanner = num_sub_groups_local % VL == 0;

    using _FirstKernel = /*TODO: oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<*/
        __two_pass_scan_kernel1<_KernelName, ValueType, BinaryOp>;

    using _SecondKernel = /*TODO: oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<*/
        __two_pass_scan_kernel2<_KernelName, ValueType, BinaryOp>;

    size_t M = __in_rng.size();
    size_t num_remaining = M;

    static_assert(oneapi::dpl::unseq_backend::__has_known_identity<BinaryOp, ValueType>::value,
                  "The prototype currently supports only known identity operators + init type");
    constexpr ValueType identity = oneapi::dpl::unseq_backend::__known_identity<BinaryOp, ValueType>;

    auto mScanLength = M;
    constexpr int J_max = 128;
    std::size_t MAX_INPUTS_PER_BLOCK = work_group_size * J_max * num_work_groups; // empirically determined for reduce_then_scan
    // items per PVC hardware thread
    int K = mScanLength >= MAX_INPUTS_PER_BLOCK
                ? MAX_INPUTS_PER_BLOCK / num_sub_groups_global
                : std::max(std::size_t(VL),
                           oneapi::dpl::__internal::__dpl_bit_ceil(num_remaining) / num_sub_groups_global);
    // SIMD vectors per PVC hardware thread
    int J = K / VL;
    int j;

    auto blockSize = (M < MAX_INPUTS_PER_BLOCK) ? M : MAX_INPUTS_PER_BLOCK;
    auto numBlocks = M / blockSize + (M % blockSize != 0);

    auto globalRange = range<1>(num_work_groups * work_group_size);
    auto localRange = range<1>(work_group_size);
    nd_range<1> range(globalRange, localRange);

    // Each element is a partial result from a subgroup. The last element is to support in-place
    // exclusive scans where we need to store the original input's last element for future use
    // before it is overwritten.
    ValueType* tmp_storage = sycl::malloc_device<ValueType>(num_sub_groups_global + 1, q);
    sycl::event event;

    // run scan kernels for all input blocks in the current buffer
    // e.g., scan length 2^24 / MAX_INPUTS_PER_BLOCK = 2 blocks
    for (std::size_t b = 0; b < numBlocks; b++)
    {
        bool is_full_block = J == J_max;
        // the first kernel computes sub-group local prefix scans and
        // subgroup local carries, one per thread
        // intermediate partial sums and carries write back to the output buffer
        event = q.submit([&](handler& h) {
            sycl::local_accessor<ValueType> sub_group_partials(num_sub_groups_local, h);
            h.depends_on(event);
            oneapi::dpl::__ranges::__require_access(h, __in_rng, __out_rng);
            h.parallel_for<_FirstKernel>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
                auto id = ndi.get_global_id(0);
                auto lid = ndi.get_local_id(0);
                auto g = ndi.get_group(0);
                auto sub_group = ndi.get_sub_group();
                auto sub_group_id = sub_group.get_group_linear_id();
                auto sub_group_local_id = sub_group.get_local_linear_id();

                ValueType v = identity;
                ValueType sub_group_carry = identity;

                size_t start_idx = (b * blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K);
                bool is_full_thread = start_idx + J * VL <= M;
                // adjust for lane-id
                start_idx += sub_group_local_id;
                // compute sub-group local pfix on T0..63, K samples/T, send to accumulator kernel
                if (is_full_thread && is_full_block)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J_max; j++)
                    {
                        v = unary_op(__in_rng[start_idx + j * VL]);
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                    }
                }
                else if (is_full_thread)
                {
                    for (int j = 0; j < J; j++)
                    {
                        v = unary_op(__in_rng[start_idx + j * VL]);
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                    }
                }
                else
                {
                    for (int j = 0; j < J; j++)
                    {
                        auto offset = start_idx + j * VL;
                        // Pass through identity if we are past the max range
                        v = offset < M ? unary_op(__in_rng[start_idx + j * VL]) : identity;
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                    }
                }

                if (sub_group_local_id == 0)
                    sub_group_partials[sub_group_id] = sub_group_carry;

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
                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(num_sub_groups_local, VL);
                    sub_group_carry = identity;
                    if (is_full_carry_scanner)
                    {
                        for (std::uint8_t i = 0; i < iters; i++)
                        {
                            v = sub_group_partials[i * VL + sub_group_local_id];
                            std::tie(v, sub_group_carry) = sub_group_scan<VL, true>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                            tmp_storage[start_idx + i * VL + sub_group_local_id] = v;
                        }
                    }
                    else
                    {
                        // In practice iters is usually 1 here when the number of sub-groups is less than the vector length.
                        // An exception would be the unlikely case where the sub-group size does not divide the work-group size
                        std::uint8_t i = 0;
                        for (; i < iters - 1; i++)
                        {
                            v = sub_group_partials[sub_group_local_id];
                            std::tie(v, sub_group_carry) = sub_group_scan<VL, true>(sub_group, std::move(v), binary_op,
                                                                                    std::move(sub_group_carry));
                            tmp_storage[start_idx + i * VL + sub_group_local_id] = v;
                        }
                        // If we are past the input range, then the previous value of v is passed to the sub-group scan.
                        // It does not affect the result as our sub_group_scan will use a mask to only process in-range elements.
                        if (i * VL + sub_group_local_id < num_sub_groups_local)
                            v = sub_group_partials[sub_group_local_id];
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, true>(
                            sub_group, std::move(v), binary_op, std::move(sub_group_carry), num_sub_groups_local);
                        if (i * VL + sub_group_local_id < num_sub_groups_local)
                            tmp_storage[start_idx + i * VL + sub_group_local_id] = v;
                    }
                }
            });
        });

        // the second kernel computes a prefix sum on sub-group local carries
        // then propogates carries inter-WG to generate thread-local versions
        // of the global carries on each sub-group; then the ouput stage
        // recomputes the thread-local prefix scan - add carry - store to compute the final sum
        event = q.submit([&](handler& CGH) {
            sycl::local_accessor<ValueType> sub_group_partials(num_sub_groups_local + 1, CGH);
            CGH.depends_on(event);
            oneapi::dpl::__ranges::__require_access(CGH, __in_rng, __out_rng);
            CGH.parallel_for<_SecondKernel>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
                auto id = ndi.get_global_id(0);
                auto lid = ndi.get_local_id(0);
                auto g = ndi.get_group(0);
                auto sub_group = ndi.get_sub_group();
                auto sub_group_id = sub_group.get_group_linear_id();
                auto sub_group_local_id = sub_group.get_local_linear_id();

                ValueType v = identity;
                ValueType carry_last = identity;

                // propogate carry in from previous block
                ValueType sub_group_carry;
                if (lid == 0)
                {
                    if (b == 0)
                        sub_group_carry = init;
                    else
                    {
                        if constexpr (Inclusive)
                            sub_group_carry = __out_rng[b * blockSize - 1];
                        else // The last block wrote an exclusive result, so we must make it inclusive.
                        {
                            // Grab the last element from the previous block that has been cached in temporary
                            // storage in the first kernel.
                            ValueType last_block_element = unary_op(tmp_storage[num_sub_groups_global]);
                            sub_group_carry = binary_op(__out_rng[b * blockSize - 1], last_block_element);
                        }
                    }
                }
                sub_group_carry = sycl::group_broadcast(ndi.get_group(), sub_group_carry, 0);

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
                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(num_sub_groups_local, VL);
                    auto csrc = g * num_sub_groups_local;
                    if (is_full_carry_scanner)
                    {
                        for (std::uint8_t i = 0; i < iters; i++)
                        {
                            sub_group_partials[i * VL + sub_group_local_id] =
                                tmp_storage[csrc + i * VL + sub_group_local_id];
                        }
                    }
                    else
                    {
                        std::uint8_t i = 0;
                        for (; i < iters - 1; i++)
                        {
                            sub_group_partials[i * VL + sub_group_local_id] =
                                tmp_storage[csrc + i * VL + sub_group_local_id];
                        }
                        if (i * VL + sub_group_local_id < num_sub_groups_local)
                        {
                            sub_group_partials[i * VL + sub_group_local_id] =
                                tmp_storage[csrc + i * VL + sub_group_local_id];
                        }
                    }

                    // step 2) load 32, 64, 96, etc. work-group carry outs on every work-group; then
                    //         compute the prefix in a sub-group to get global work-group carries
                    //         memory accesses: gather(63, 127, 191, 255, ...)
                    uint32_t offset = num_sub_groups_local - 1;
                    ValueType carry = identity;
                    _ONEDPL_PRAGMA_UNROLL
                    // only need 32 carries for WGs0..WG32, 64 for WGs32..WGs64, etc.
                    for (int i = 0; i < (g >> log2_VL) + 1; i++)
                    {
                        v = tmp_storage[i * num_sub_groups_local * VL +
                                        (num_sub_groups_local * sub_group_local_id + offset)];
                        std::tie(v, carry) = sub_group_scan<VL, true>(sub_group, std::move(v), binary_op, carry);
                        if (i != (g >> log2_VL))
                            carry_last = carry;
                    }
                }
                // For the exclusive scan case:
                // While the first sub-group is doing work, have the last item in the group store the last element
                // in the block to temporary storage for use in the next block.
                // This is required to support in-place exclusive scans as the input values will be overwritten.
                if constexpr (!Inclusive)
                {
                    auto global_id = ndi.get_global_linear_id();
                    if (global_id == num_work_groups * work_group_size - 1)
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
                    ValueType adj_work_group_carry;
                    auto carry_offset = 0;
                    if (g % VL != 0)
                    {
                        adj_work_group_carry = sycl::group_broadcast(sub_group, v, g % VL - 1);
                    }
                    else
                    {
                        adj_work_group_carry = carry_last;
                    }
                    std::uint8_t iters = oneapi::dpl::__internal::__dpl_ceiling_div(num_sub_groups_local, VL);
                    if (is_full_carry_scanner)
                    {
                        for (std::uint8_t i = 0; i < iters; ++i)
                        {
                            sub_group_partials[carry_offset + sub_group_local_id] =
                                binary_op(adj_work_group_carry, sub_group_partials[carry_offset + sub_group_local_id]);
                            carry_offset += VL;
                        }
                    }
                    else
                    {
                        std::uint8_t i = 0;
                        for (; i < iters - 1; ++i)
                        {
                            sub_group_partials[carry_offset + sub_group_local_id] =
                                binary_op(adj_work_group_carry, sub_group_partials[carry_offset + sub_group_local_id]);
                            carry_offset += VL;
                        }
                        if (i * VL + sub_group_local_id < num_sub_groups_local)
                        {
                            sub_group_partials[carry_offset + sub_group_local_id] =
                                binary_op(adj_work_group_carry, sub_group_partials[carry_offset + sub_group_local_id]);
                            carry_offset += VL;
                        }
                    }
                    if (sub_group_local_id == 0)
                        sub_group_partials[num_sub_groups_local] = adj_work_group_carry;
                }
                //sycl::group_barrier(ndi.get_group());
                ndi.barrier(sycl::access::fence_space::local_space);

                // Get inter-work group and adjusted for intra-work group prefix
                if (sub_group_id > 0)
                {
                    if (sub_group_local_id == 0)
                    {
                        sub_group_carry = binary_op(sub_group_carry, sub_group_partials[sub_group_id - 1]);
                    }
                    sub_group_carry = sycl::group_broadcast(sub_group, sub_group_carry, 0);
                }
                else if (g > 0)
                {
                    if (sub_group_local_id == 0)
                    {
                        sub_group_carry = binary_op(sub_group_carry, sub_group_partials[num_sub_groups_local]);
                    }
                    sub_group_carry = sycl::group_broadcast(sub_group, sub_group_carry, 0);
                }

                // step 5) apply global carries
                size_t start_idx = (b * blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K);
                bool is_full_thread = start_idx + J * VL <= M;
                start_idx += sub_group_local_id;
                if (is_full_thread && is_full_block)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J_max; j++)
                    {
                        v = unary_op(__in_rng[start_idx + j * VL]);
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                        __out_rng[start_idx + j * VL] = v;
                    }
                }
                else if (is_full_thread)
                {
                    for (int j = 0; j < J; j++)
                    {
                        v = unary_op(__in_rng[start_idx + j * VL]);
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                        __out_rng[start_idx + j * VL] = v;
                    }
                }
                else
                {
                    for (int j = 0; j < J; j++)
                    {
                        auto offset = start_idx + j * VL;
                        // Pass through identity if we are past the max range
                        v = offset < M ? unary_op(__in_rng[offset]) : identity;
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        std::tie(v, sub_group_carry) = sub_group_scan<VL, Inclusive>(sub_group, std::move(v), binary_op, std::move(sub_group_carry));
                        if (offset < M)
                            __out_rng[offset] = v;
                    }
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

namespace ranges {

template <typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp, typename UnaryOp, typename ValueType>
void
two_pass_transform_exclusive_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng,
                        BinaryOp binary_op, UnaryOp unary_op, ValueType init)
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    __impl::two_pass_scan<false, _KernelName>(q, std::move(__in_view), std::move(__out_view), binary_op, unary_op, init);
}

template <typename _KernelName, typename _InRng, typename _OutRng, typename BinaryOp,  typename UnaryOp, typename ValueType>
void
two_pass_transform_inclusive_scan(sycl::queue q, _InRng&& __in_rng, _OutRng&& __out_rng,
                        BinaryOp binary_op, UnaryOp unary_op, ValueType init)
{
    auto __in_view = oneapi::dpl::__ranges::views::all(std::forward<_InRng>(__in_rng));
    auto __out_view = oneapi::dpl::__ranges::views::all(std::forward<_OutRng>(__out_rng));

    __impl::two_pass_scan<true, _KernelName>(q, std::move(__in_view), std::move(__out_view), binary_op, unary_op, init);
}

} // namespace ranges

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator, typename OutputIterator, typename BinaryOp, typename UnaryOp, typename ValueType>
void
two_pass_transform_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                                  BinaryOp binary_op, UnaryOp unary_op, ValueType init)
{
    auto __n = last - first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, InputIterator>();
    auto __buf1 = __keep1(first, last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, OutputIterator>();
    auto __buf2 = __keep2(result, result + __n);

    ranges::two_pass_transform_inclusive_scan<_KernelName>(q, __buf1.all_view(), __buf2.all_view(), binary_op, unary_op, init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator, typename OutputIterator, typename ValueType, typename BinaryOp, typename UnaryOp>
void
two_pass_transform_exclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        ValueType init, BinaryOp binary_op, UnaryOp unary_op)
{
    auto __n = last - first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, InputIterator>();
    auto __buf1 = __keep1(first, last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, OutputIterator>();
    auto __buf2 = __keep2(result, result + __n);

    ranges::two_pass_transform_exclusive_scan<_KernelName>(q, __buf1.all_view(), __buf2.all_view(), binary_op, unary_op, init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator, typename OutputIterator, typename BinaryOp, typename ValueType>
void
two_pass_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        BinaryOp binary_op, ValueType init)
{
    two_pass_transform_inclusive_scan<_KernelName>(q, first, last, result, binary_op, oneapi::dpl::__internal::__no_op(), init);
}

template <typename _KernelName = oneapi::dpl::execution::DefaultKernelName, typename InputIterator, typename OutputIterator, typename BinaryOp, typename ValueType>
void
two_pass_exclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        ValueType init, BinaryOp binary_op)
{
    two_pass_transform_exclusive_scan<_KernelName>(q, first, last, result, init, binary_op, oneapi::dpl::__internal::__no_op());
}

} // namespace oneapi::dpl::experimental::kt::gpu
