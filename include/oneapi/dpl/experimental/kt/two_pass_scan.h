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

#include "internal/esimd_defs.h"
#include "../../pstl/utils.h"
#include "../../pstl/hetero/dpcpp/unseq_backend_sycl.h"

namespace oneapi::dpl::experimental::kt::gpu
{

namespace
{
template <std::uint8_t VL, typename SubGroup, typename BinaryOp, typename ValueType>
auto
sub_group_inclusive_scan(const SubGroup& sub_group, ValueType value, BinaryOp binary_op, const ValueType& init)
{
    std::uint8_t sub_group_local_id = sub_group.get_local_linear_id();
    _ONEDPL_PRAGMA_UNROLL
    for (std::uint8_t shift = 1; shift <= VL / 2; shift <<= 1)
    {
        auto partial_carry_in = sycl::shift_group_right(sub_group, value, shift);
        if (sub_group_local_id >= shift)
        {
            value = binary_op(partial_carry_in, value);
        }
    }
    value = binary_op(init, value);
    return std::move(value);
}
} // namespace

// Named two_pass_scan for now to avoid name clash with single pass KT
template <typename InputIterator, typename OutputIterator, typename BinaryOp, typename ValueType>
void
two_pass_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result,
                        BinaryOp binary_op, ValueType init)
{
    using namespace sycl;

    // PVC 1 tile
    constexpr std::uint32_t log2_VL = 5;
    constexpr std::uint32_t VL = 1 << log2_VL;               // simd vector length 2^5 = 32
    constexpr std::uint32_t MAX_INPUTS_PER_BLOCK = 16777216; // empirically determined for reduce_then_scan

    constexpr std::uint32_t work_group_size = 1024;
    constexpr std::uint32_t num_work_groups = 128;
    constexpr std::uint32_t num_sub_groups_local = work_group_size / VL;
    constexpr std::uint32_t num_sub_groups_global = num_sub_groups_local * num_work_groups;

    uint32_t M = std::distance(first, last);
    uint32_t num_remaining = M;

    static_assert(oneapi::dpl::unseq_backend::__has_known_identity<BinaryOp, ValueType>::value,
                  "The prototype currently supports only known identity operators + init type");
    constexpr ValueType identity = oneapi::dpl::unseq_backend::__known_identity<BinaryOp, ValueType>;

    auto mScanLength = M;
    // items per PVC hardware thread
    int K = mScanLength >= MAX_INPUTS_PER_BLOCK
                ? MAX_INPUTS_PER_BLOCK / num_sub_groups_global
                : std::max(std::uint32_t(VL),
                           oneapi::dpl::__internal::__dpl_bit_ceil(num_remaining) / num_sub_groups_global);
    // SIMD vectors per PVC hardware thread
    int J = K / VL;
    int j;

    auto blockSize = (M < MAX_INPUTS_PER_BLOCK) ? M : MAX_INPUTS_PER_BLOCK;
    auto numBlocks = M / blockSize + (M % blockSize != 0);

    auto globalRange = range<1>(num_work_groups * work_group_size);
    auto localRange = range<1>(work_group_size);
    nd_range<1> range(globalRange, localRange);

    // Each element is a partial result from a subgroup
    ValueType* tmp_storage = sycl::malloc_device<ValueType>(num_sub_groups_global, q);
    sycl::event event;

    // run scan kernels for all input blocks in the current buffer
    // e.g., scan length 2^24 / MAX_INPUTS_PER_BLOCK = 2 blocks
    for (int b = 0; b < numBlocks; b++)
    {
        // the first kernel computes sub-group local prefix scans and
        // subgroup local carries, one per thread
        // intermediate partial sums and carries write back to the output buffer
        event = q.submit([&](handler& h) {
            sycl::local_accessor<ValueType> sub_group_partials(num_sub_groups_local, h);
            h.depends_on(event);
            h.parallel_for<class kernel1>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
                auto id = ndi.get_global_id(0);
                auto lid = ndi.get_local_id(0);
                auto g = ndi.get_group(0);
                auto sub_group = ndi.get_sub_group();
                auto sub_group_id = sub_group.get_group_linear_id();
                auto sub_group_local_id = sub_group.get_local_linear_id();

                ValueType v = identity;
                ValueType sub_group_carry = identity;

                uint32_t start_idx = (b * blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K);
                bool is_full_thread = start_idx + J * VL <= M;
                // adjust for lane-id
                start_idx += sub_group_local_id;
                // compute sub-group local pfix on T0..63, K samples/T, send to accumulator kernel
                if (is_full_thread)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J; j++)
                    {
                        v = first[start_idx + j * VL];
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, sub_group_carry);
                        // Last sub-group lane communicates its carry to everyone else in the sub-group
                        sub_group_carry = sycl::group_broadcast(sub_group, v, VL - 1);
                    }
                }
                else
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J; j++)
                    {
                        auto offset = start_idx + j * VL;
                        // Pass through identity if we are past the max range
                        v = offset < M ? first[start_idx + j * VL] : identity;
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, sub_group_carry);
                        // Last sub-group lane communicates its carry to everyone else in the sub-group
                        sub_group_carry = sycl::group_broadcast(sub_group, v, VL - 1);
                    }
                }

                if (sub_group_local_id == VL - 1)
                    sub_group_partials[sub_group_id] = v;

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
                    // TODO: handle if num_sub_groups_local < VL
                    constexpr std::uint8_t iters = num_sub_groups_local / VL;
                    sub_group_carry = identity;
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint8_t i = 0; i < iters; i++)
                    {
                        v = sub_group_partials[i * VL + sub_group_local_id];
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, sub_group_carry);
                        tmp_storage[start_idx + i * VL + sub_group_local_id] = v;
                        sub_group_carry = sycl::group_broadcast(sub_group, v, VL - 1);
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
            CGH.parallel_for<class kernel2>(range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
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
                        sub_group_carry = result[b * blockSize - 1];
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
                    constexpr std::uint8_t iters = num_sub_groups_local / VL;
                    auto csrc = g * num_sub_groups_local;
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint8_t i = 0; i < iters; i++)
                    {
                        sub_group_partials[i * VL + sub_group_local_id] =
                            tmp_storage[csrc + i * VL + sub_group_local_id];
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
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, carry);
                        carry = sycl::group_broadcast(sub_group, v, VL - 1);
                        if (i != (g >> log2_VL))
                            carry_last = carry;
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
                    constexpr std::uint8_t iters = num_sub_groups_local / VL;
                    for (std::uint8_t i = 0; i < iters; ++i)
                    {
                        sub_group_partials[carry_offset + sub_group_local_id] =
                            binary_op(adj_work_group_carry, sub_group_partials[carry_offset + sub_group_local_id]);
                        carry_offset += VL;
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
                uint32_t start_idx = (b * blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K);
                bool is_full_thread = start_idx + J * VL <= M;
                start_idx += sub_group_local_id;
                if (is_full_thread)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J; j++)
                    {
                        v = first[start_idx + j * VL];
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, sub_group_carry);
                        result[start_idx + j * VL] = v;
                        // Last sub-group lane communicates its carry to everyone else in the sub-group
                        sub_group_carry = sycl::group_broadcast(sub_group, v, VL - 1);
                    }
                }
                else
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (int j = 0; j < J; j++)
                    {
                        auto offset = start_idx + j * VL;
                        // Pass through identity if we are past the max range
                        v = offset < M ? first[offset] : identity;
                        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
                        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), binary_op, sub_group_carry);
                        if (offset < M)
                            result[offset] = v;
                        // Last sub-group lane communicates its carry to everyone else in the sub-group
                        sub_group_carry = sycl::group_broadcast(sub_group, v, VL - 1);
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
                    : std::max(std::uint32_t(VL),
                               oneapi::dpl::__internal::__dpl_bit_ceil(num_remaining) / num_sub_groups_global);
            // SIMD vectors per PVC hardware thread
            J = K / VL;
        }
    } // block
    event.wait();
    sycl::free(tmp_storage, q);
}

} // namespace oneapi::dpl::experimental::kt::gpu
