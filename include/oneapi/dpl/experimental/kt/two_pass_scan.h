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

#include "internal/esimd_defs.h"

namespace oneapi::dpl::experimental::kt::gpu
{

namespace {
template <std::uint8_t VL, typename SubGroup, typename ValueType>
auto
sub_group_inclusive_scan(const SubGroup& sub_group, ValueType value, const ValueType& init)
{
    std::uint8_t sub_group_local_id = sub_group.get_local_linear_id();
    #pragma unroll
    for (std::uint8_t shift = 1; shift <= VL / 2; shift <<= 1)
    {
        auto tmp = sycl::shift_group_right(sub_group, value, shift);
        if (sub_group_local_id - shift >= 0)
        {
            value += tmp;
        }
    }
    value += init;
    return std::move(value);
}
}

// Named two_pass_scan for now to avoid name clash with single pass KT
template <typename InputIterator, typename OutputIterator, typename UNUSED1, typename UNUSED2>
void two_pass_inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result, UNUSED1, UNUSED2)
{
    using namespace sycl;
    using namespace sycl::ext::intel;
    using namespace sycl::ext::intel::esimd;

    using ValueType = typename std::iterator_traits<InputIterator>::value_type;

    // PVC 1 tile
    constexpr std::uint32_t VL                   = 32;         // simd vector length
    constexpr std::uint32_t MAX_INPUTS_PER_BLOCK = 16777216;   // empirically determined for reduce_then_scan

    // export NEOReadDebugKeys=1
    // we need export OverrideMaxWorkgroupSize=2048
    constexpr std::uint32_t work_group_size = 2048; // TODO: experiment with 1024. largest supported without experimental driver macros
    constexpr std::uint32_t num_work_groups = 64;   // If above is 1024, then this should be 128 for full saturation of PVC 1-tile 
    constexpr std::uint32_t num_sub_groups_local = work_group_size / VL;   // If above is 1024, then this should be 128 for full saturation of PVC 1-tile
    constexpr std::uint32_t num_sub_groups_global = num_sub_groups_local * num_work_groups;

    int M = std::distance(first, last);
    auto mScanLength = M;
    // items per PVC hardware thread
    int K = mScanLength >= MAX_INPUTS_PER_BLOCK ? MAX_INPUTS_PER_BLOCK / num_sub_groups_global : mScanLength / num_sub_groups_global;
    // SIMD vectors per PVC hardware thread
    int J = K/VL;
    int j;

    // block inputs according to slm capacity
    // such that one block fills slm across the pvc tile
    // in the current implementaiton constrained by lack of esimd support for global
    // barriers (root_group) and device memory global atomics 2 kernels are used
    // for synchronization instead of a single kernel with ineternal sync and carry prop
    // so slm capacity is not a limiting factor, but the merged kernel
    // version will rely on blocking to fully utilize slm
    // for scan lengths less than 2^23 block size is reduced accordingly
    auto blockSize = ( M < MAX_INPUTS_PER_BLOCK ) ? M : MAX_INPUTS_PER_BLOCK;
    auto numBlocks = M / blockSize;

    auto globalRange = range<1>(num_work_groups * work_group_size);
    auto localRange = range<1>(work_group_size);
    nd_range<1> range(globalRange, localRange);

    // Each element is a partial result from a subgroup
    ValueType *tmp_storage = sycl::malloc_device<ValueType>(num_sub_groups_global, q);
    sycl::event event;

    // run scan kernels for all input blocks in the current buffer
    // e.g., scan length 2^24 / MAX_INPUTS_PER_BLOCK = 2 blocks
    for ( int b=0; b<numBlocks; b++ ) {
    // the first kernel computes thread-local prefix scans and 
    // thread local carries, one per thread
    // intermediate partial sums and carries write back to the output buffer
    event = q.submit([&](handler &h) {
    sycl::local_accessor<ValueType> sub_group_partials(num_sub_groups_local, h);
    h.depends_on(event);
    h.parallel_for<class kernel1>( range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
        auto id = ndi.get_global_id(0);
        auto lid = ndi.get_local_id(0);
        auto g = ndi.get_group(0);
        auto sub_group = ndi.get_sub_group();
        auto sub_group_id = sub_group.get_group_linear_id();
        auto sub_group_local_id = sub_group.get_local_linear_id();
        uint32_t start_idx = (b*blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K) + sub_group_local_id;
        ValueType v = 0;
        ValueType init = 0;
        
        // compute thread-local pfix on T0..63, K samples/T, send to accumulator kernel
        #pragma unroll
        for ( int j=0; j<J; j++ ) {
        v = first[start_idx + j*VL];
        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), init);
        // Last sub-group lane communicates its carry to everyone else in the sub-group
        init = sycl::group_broadcast(sub_group, v, VL-1);
        }

        if (sub_group_local_id == VL - 1)
            sub_group_partials[sub_group_id] = v;

        // TODO: This is slower then ndi.barrier which was removed in SYCL2020. Can we do anything about it?
        //sycl::group_barrier(ndi.get_group());
        ndi.barrier(sycl::access::fence_space::local_space);

        // compute thread-local prefix sums on (T0..63) carries
        // and store to scratch space at the end of dst; next
        // accumulator kernel takes M thread carries from scratch
        // to compute a prefix sum on global carries
        if ( sub_group_id == 0 ) {
        start_idx = (g * num_sub_groups_local);
        // TODO: handle if num_sub_groups_local < VL
        constexpr std::uint8_t iters = num_sub_groups_local / VL;
        init = 0;
        #pragma unroll
        for( std::uint8_t i=0; i<iters; i++ ) {
            v = sub_group_partials[i*VL + sub_group_local_id];
            v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), init);
            tmp_storage[start_idx + i*VL + sub_group_local_id] = v;
            init = sycl::group_broadcast(sub_group, v, VL-1);
        }
        }
    });
    });
    
    // the second kernel computes a prefix sum on thread-local carries
    // then propogates carries inter-Xe to generate thread-local versions
    // of the global carries on each Xe thread; then the ouput stage
    // does load - add carry - store to compute the final sum
    // N.B. the two kernels will be merged once inter-Xe sync has been 
    // fixed in ESIMD.  Currently atomics and global device memory are 
    // not coherent inter-Xe
    event = q.submit([&](handler &CGH) {
    sycl::local_accessor<ValueType> sub_group_partials(num_sub_groups_local + 1, CGH);
    CGH.depends_on(event);
    CGH.parallel_for<class kernel2>( range, [=](nd_item<1> ndi) [[sycl::reqd_sub_group_size(VL)]] {
        auto id = ndi.get_global_id(0);
        auto lid = ndi.get_local_id(0);
        auto g = ndi.get_group(0);
        auto sub_group = ndi.get_sub_group();
        auto sub_group_id = sub_group.get_group_linear_id();
        auto sub_group_local_id = sub_group.get_local_linear_id();

        ValueType v;
        ValueType carry31;

        // propogate carry in from previous block
        ValueType carry_in = 0;
        if (b > 0 && lid == 0)
            carry_in = result[b*blockSize-1];
        carry_in = sycl::group_broadcast(ndi.get_group(), carry_in, 0);

        // on each Xe T0: 
        // 1. load 64 T-local carry pfix sums (T0..63) to slm
        // 2. load 32 or 64 T63 Xe carry-outs (32 for Xe num<32, 64 above),
        //    and then compute the prefix sum to generate global carry out
        //    for each Xe, i.e., prefix sum on T63 carries over all Xe.
        // 3. on each Xe select theh adjacent neighboring Xe carry in
        // 4. on each Xe add the global carry-in to 64 T-local pfix sums to  
        //    get a T-local global carry in
        // 5. load T-local pfix values from memory, add the T-local global carries, 
        //    and then write back the final values to memory
        if ( sub_group_id == 0 ) {
        // step 1) load to Xe slm the Xe-local 64 prefix sums 
        //         on 64 T-local carries 
        //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
        //           63: sum(T0 carry...T63 carry)
        constexpr std::uint8_t iters = num_sub_groups_local / VL;
        auto csrc = g*num_sub_groups_local;
        #pragma unroll
        for(std::uint8_t i=0;i<iters;i++) {
            sub_group_partials[i*VL + sub_group_local_id] = tmp_storage[csrc + i*VL + sub_group_local_id];
        } 

        // step 2) load 32 or 64 Xe carry outs on every Xe; then
        //         compute the prefix to get global Xe carries
        //         v = gather(63, 127, 191, 255, ...)
        uint32_t offset = num_sub_groups_local - 1;
        ValueType init = 0;
        #pragma unroll
        // only need 32 carries for Xe0..Xe31
        for( int i=0; i<(g>>5)+1; i++ ) {
            v = tmp_storage[i*num_sub_groups_local*VL + (num_sub_groups_local * sub_group_local_id + offset)];
            v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), init);
            init = sycl::group_broadcast(sub_group, v, VL-1);
            if (i==0) carry31 = init;
        }
        }

        // wait for step 2 to complete on all Xe, i.e.,
        // for all Xe to finish computing local vector(s) of global carries
        // N.B. barrier could be earlier, guarantees slm local carry update
        //sycl::group_barrier(ndi.get_group());
        ndi.barrier(sycl::access::fence_space::local_space);

        // steps 3/4) load global carry in from Xe neighbor 
        //            and apply to slm local prefix carries T0..T63 
        if ( (sub_group_id==0) && (g>0) ) {
        ValueType carry;
        auto carry_offset = 0;
        if (g!=32) {
            carry = sycl::group_broadcast(sub_group, v, g%32-1);
        } else {
            carry = carry31;
        }
        // TODO: this seems hardcoded for 64 sub groups per work group
        sub_group_partials[carry_offset + sub_group_local_id] = sub_group_partials[carry_offset + sub_group_local_id] + carry;
        carry_offset += VL;
        sub_group_partials[carry_offset + sub_group_local_id] = sub_group_partials[carry_offset + sub_group_local_id] + carry;
        if (sub_group_local_id == 0)
            sub_group_partials[num_sub_groups_local] = carry;
        }
        //sycl::group_barrier(ndi.get_group());
        ndi.barrier(sycl::access::fence_space::local_space);

        // get global carry adjusted for thread-local prefix
        if ( sub_group_id > 0 )
        {
            if (sub_group_local_id == 0)
            {
                carry_in += sub_group_partials[sub_group_id - 1];
            }
            carry_in = sycl::group_broadcast(sub_group, carry_in, 0);
        }
        else if (g > 0)
        {
            if (sub_group_local_id == 0)
            {
                carry_in += sub_group_partials[num_sub_groups_local];
            }
            carry_in = sycl::group_broadcast(sub_group, carry_in, 0);
        }

        // step 5) apply global carries
        // N.B. grouping loads and stores and using more registers
        // e.g., load load load load v1+c v2c v3+c v4+c store store store store
        // for simd carry ops has shown +5-10% perforamnce gain on PVC
        // but results were inconsistent and the ESIMD design team guidance
        // was to use block_load / _store without grouping
        // even through the data so far does not agree with the guidance but for simplicity
        // with variable scan lengths grouping is not used currently
        // grouping and prefetch are likely to improve throughput in future versions
        uint32_t start_idx = (b*blockSize) + (g * K * num_sub_groups_local) + (sub_group_id * K) + sub_group_local_id;
        ValueType init = carry_in;
        #pragma unroll
        for ( int j=0; j<J; j++ ) {
        v = first[start_idx + j*VL];
        // In principle we could use SYCL group scan. Stick to our own for now for full control of implementation.
        v = sub_group_inclusive_scan<VL>(sub_group, std::move(v), init);
        result[start_idx + j*VL] = v;
        // Last sub-group lane communicates its carry to everyone else in the sub-group
        init = sycl::group_broadcast(sub_group, v, VL-1);
        }
    });
    });

    } // block
    event.wait();
    sycl::free(tmp_storage, q);
}

} // namespace oneapi::dpl::experimental::kt
