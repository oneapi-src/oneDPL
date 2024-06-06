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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/builtins_esimd.hpp>
#include <time.h>
#include <sycl/ext/intel/esimd.hpp>
#include <iterator>

#include "internal/esimd_defs.h"


#define prefix_sum( v, c, z, t ) \
  t.template select<31,1>(1) = v.template select<31,1>(0); \
  v = v + t; t = z; \
  t.template select<30,1>(2) = v.template select<30,1>(0); \
  v = v + t; t = z; \
  t.template select<28,1>(4) = v.template select<28,1>(0); \
  v = v + t; t = z; \
  t.template select<24,1>(8) = v.template select<24,1>(0); \
  v = v + t; t = z; \
  t.template select<16,1>(16) = v.template select<16,1>(0); \
  v = v + t + c; t = z; \
  c = v.template replicate_w<VL,1>(31);

namespace oneapi::dpl::experimental::kt::gpu::esimd
{

template <typename InputIterator, typename OutputIterator, typename UNUSED1, typename UNUSED2>
void inclusive_scan(sycl::queue q, InputIterator first, InputIterator last, OutputIterator result, UNUSED1, UNUSED2)
{
    using namespace sycl;
    using namespace sycl::ext::intel;
    using namespace sycl::ext::intel::esimd;

    using ValueType = typename std::iterator_traits<InputIterator>::value_type;

    // PVC 1 tile
    constexpr std::uint32_t VL                   = 32;         // simd vector length
    constexpr std::uint32_t NUM_THREADS_GLOBAL   = 4096;       // threads per tile
    constexpr std::uint32_t NUM_THREADS_LOCAL    = 64;         // threads per Xe
    constexpr std::uint32_t NUM_XE               = 64;         // number of Xe per tile
    constexpr std::uint32_t MAX_SLM_BYTES_PER_XE = (512*1024); // per opt guide 512, actual 128?
    constexpr std::uint32_t MAX_INPUTS_PER_BLOCK = 16777216;   // empirically determined for reduce_then_scan

    int M = std::distance(first, last);
    auto mScanLength = M;
    // items per PVC hardware thread
    int K = mScanLength >= MAX_INPUTS_PER_BLOCK ? MAX_INPUTS_PER_BLOCK / NUM_THREADS_GLOBAL : mScanLength / NUM_THREADS_GLOBAL;
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

    auto globalRange = range<1>(NUM_THREADS_GLOBAL);
    auto localRange = range<1>(NUM_THREADS_LOCAL);
    nd_range<1> range(globalRange, localRange);

    ValueType *tmp_storage = sycl::malloc_device<ValueType>(NUM_THREADS_GLOBAL, q);

    // run scan kernels for all input blocks in the current buffer
    // e.g., scan length 2^24 / MAX_INPUTS_PER_BLOCK = 2 blocks 
    for ( int b=0; b<numBlocks; b++ ) {
    int offset = 0; // TODO remove

    // the first kernel computes thread-local prefix scans and 
    // thread local carries, one per thread
    // intermediate partial sums and carries write back to the output buffer
    q.submit([&](handler &h) {
    //auto src = bInputData.template get_access<access::mode::read>(h);
    //auto dst = bOutputData.template get_access<access::mode::read_write>(h);
    h.parallel_for( range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
        //auto offset = j * M * sizeof(uint32_t);
        slm_init<1024>();
        auto id = ndi.get_global_id(0);
        auto lid = ndi.get_local_id(0);
        auto g = ndi.get_group(0);
        auto addr = id * K * sizeof(ValueType) + offset + (b*blockSize*sizeof(ValueType));
        // N.B. select() operator is broken for uint_32t, replace everywhere with uint
        simd<ValueType,VL> c, t, v, z;
        z = c = t = 0;
        // compute thread-local pfix on T0..63, K samples/T, send to accumulator kernel
        #pragma unroll
        for ( int j=0; j<J; j++ ) {
        #if 0
        v.copy_from(first,addr);
        #else
	// We use reinterpret casting to get around pointer arithmetic jumping by the size of the
	// underlying type instead of individual bytes. The buffer accessor ESIMD call requires an offset in bytes.
	// During productization this should be better handled.
        v.copy_from(reinterpret_cast<ValueType*>(reinterpret_cast<uint8_t*>(first) + addr));
        #endif
        prefix_sum( v, c, z, t );
        addr += (VL*sizeof(ValueType));
        }

        // store T local carry outs (T0..63) to slm
        int caddr = lid*sizeof(ValueType);
        slm_scalar_store<ValueType>(caddr,c[31]);
        __dpl_esimd::__ns::barrier();

        // compute thread-local prefix sums on (T0..63) carries
        // and store to scratch space at the end of dst; next
        // accumulator kernel takes M thread carries from scratch
        // to compute a prefix sum on global carries
        if ( lid == 0 ) {
        #if 0
        addr = (M + g*NUM_THREADS_LOCAL) * sizeof(uint) + offset;
        #else
        addr = (g*NUM_THREADS_LOCAL) * sizeof(ValueType) + offset;
        #endif
        c = z; 
        #pragma unroll
        for( int i=0; i<2; i++ ) {
            v = slm_block_load<ValueType,VL>(caddr);
            caddr += (VL*sizeof(ValueType));
            prefix_sum( v, c, z, t );
            #if 0
            v.copy_to(dst,addr);
            #else
            v.copy_to(reinterpret_cast<ValueType*>(reinterpret_cast<uint8_t*>(tmp_storage) + addr));
            #endif
            addr += (VL*sizeof(ValueType));
        }
        }
    });
    }).wait();

    // the second kernel computes a prefix sum on thread-local carries
    // then propogates carries inter-Xe to generate thread-local versions
    // of the global carries on each Xe thread; then the ouput stage
    // does load - add carry - store to compute the final sum
    // N.B. the two kernels will be merged once inter-Xe sync has been 
    // fixed in ESIMD.  Currently atomics and global device memory are 
    // not coherent inter-Xe
    q.submit([&](handler &CGH) {
    //auto srcDst = bOutputData.template get_access<access::mode::read_write>(CGH);
    CGH.parallel_for( range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
        slm_init<1024>();
        auto id = ndi.get_global_id(0);
        auto lid = ndi.get_local_id(0);
        auto g = ndi.get_group(0);
        simd<ValueType,VL> v;
        ValueType carry31;

        // propogate carry in from previous block
        ValueType carry_in = 0;
        if (b > 0)
            carry_in = gather<ValueType, 1>(result, simd<uint32_t, 1>(offset+(b*blockSize-1)*sizeof(ValueType)))[0];

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
        if ( lid == 0 ) {
        // step 1) load to Xe slm the Xe-local 64 prefix sums 
        //         on 64 T-local carries 
        //            0: T0 carry, 1: T0 + T1 carry, 2: T0 + T1 + T2 carry, ...
        //           63: sum(T0 carry...T63 carry)
        //auto csrc = (M + g*NUM_THREADS_LOCAL) * sizeof(uint) + offset;
        auto csrc = (g*NUM_THREADS_LOCAL) * sizeof(ValueType) + offset;
        auto cdst = 0;
        auto stride = VL*sizeof(ValueType);
        #pragma unroll
        for(int i=0;i<2;i++) {
            #if 0
            v.copy_from(result,csrc);
            #else
            v.copy_from(reinterpret_cast<ValueType*>(reinterpret_cast<uint8_t*>(tmp_storage) + csrc));
            #endif
            slm_block_store<ValueType,VL>(cdst,v);
            csrc += stride;
            cdst += stride;
        } 

        // step 2) load 32 or 64 Xe carry outs on every Xe; then
        //         compute the prefix to get global Xe carries
        //         v = gather(63, 127, 191, 255, ...)
        //simd<uint,VL> addr((M+NUM_THREADS_LOCAL-1)*sizeof(uint),NUM_THREADS_LOCAL*sizeof(uint));
        simd<uint32_t,VL> addr((NUM_THREADS_LOCAL-1)*sizeof(ValueType),NUM_THREADS_LOCAL*sizeof(ValueType));
        addr += offset;  
        simd<ValueType,VL> t, z, c;
        c = t = z = 0;
        #pragma unroll
        // only need 32 carries for Xe0..Xe31
        for( int i=0; i<(g>>5)+1; i++ ) {
            v = gather<ValueType,VL>(tmp_storage,addr);
            addr += (NUM_THREADS_LOCAL*VL*sizeof(ValueType));
            prefix_sum( v, c, z, t );
            if (i==0) carry31 = v[31];
        }
        }

        // wait for step 2 to complete on all Xe, i.e.,
        // for all Xe to finish computing local vector(s) of global carries
        // N.B. barrier could be earlier, guarantees slm local carry update
        __dpl_esimd::__ns::barrier();    

        // steps 3/4) load global carry in from Xe neighbor 
        //            and apply to slm local prefix carries T0..T63 
        if ( (lid==0) && (g>0) ) {
        simd<ValueType,VL> t;
        ValueType carry;
        auto caddr = 0;
        if (g!=32) {
            carry = v[g%32-1];
        } else {
            carry = carry31;
        }
        t = slm_block_load<ValueType,VL>(caddr) + carry;
        slm_block_store<ValueType,VL>(caddr,t);
        caddr += VL * sizeof(ValueType);
        t = slm_block_load<ValueType,VL>(caddr) + carry;
        slm_block_store<ValueType,VL>(caddr,t);
        slm_scalar_store<ValueType>(64*sizeof(ValueType),carry);
        }
        __dpl_esimd::__ns::barrier();

        // get global carry adjusted for thread-local prefix
        auto maddr = id * K * sizeof(ValueType) + offset + (b*blockSize*sizeof(ValueType));
        if ( lid > 0 ) {
        carry_in += slm_scalar_load<ValueType>((lid-1)*sizeof(ValueType));
        } else {
        if ( g > 0 ) {
        carry_in += slm_scalar_load<ValueType>(64*sizeof(ValueType));
        }
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
        simd<ValueType,VL> t, z, c;
        c = t = z = 0;
        #pragma unroll
        for ( int j=0; j<J; j++ ) {
        v = block_load<ValueType,VL>(first, maddr);
        prefix_sum( v, c, z, t );
        v += carry_in;
        block_store<ValueType,VL>(result, maddr, v);
        maddr += (VL*sizeof(ValueType));
        }  
    });
    }).wait(); 

    } // block
    sycl::free(tmp_storage, q);
}

} // namespace oneapi::dpl::experimental::kt
