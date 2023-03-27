// -*- C++ -*-
//===-- esimd_radix_sort_one_wg.h --------------------------------===//
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

#ifndef _ONEDPL_esimd_radix_sort_one_wg_H
#define _ONEDPL_esimd_radix_sort_one_wg_H

#include <ext/intel/esimd.hpp>
#include "../sycl_defs.h"
#include "../execution_sycl_defs.h"
#include "../parallel_backend_sycl_utils.h"
#include "../utils_ranges_sycl.h"


#include "esimd_radix_sort_utils.h"
#include "../../../utils.h"

#include <cstdint>

namespace oneapi::dpl::experimental::esimd::impl
{

template <typename KeyT, typename InputT, uint32_t RADIX_BITS, uint32_t PROCESS_SIZE>
void one_wg_kernel(sycl::nd_item<1> idx, uint32_t n, uint32_t THREAD_PER_TG, const InputT& input) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using device_addr_t = uint32_t;

    uint32_t local_tid = idx.get_local_linear_id();
    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);
    constexpr uint32_t MAX_THREAD_PER_TG = 64;
    constexpr bin_t MASK = BIN_COUNT - 1;

    constexpr uint32_t REORDER_SLM_SIZE = PROCESS_SIZE * sizeof(KeyT) * MAX_THREAD_PER_TG; // reorder buffer
    constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * MAX_THREAD_PER_TG;             // bin hist working buffer
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+1)*sizeof(hist_t);                // incoming offset buffer

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  PROCESS_SIZE = 256, BIN_COUNT = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    slm_init( std::max(REORDER_SLM_SIZE,  BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE));
    uint32_t slm_reorder_start = 0;
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_offset = slm_bin_hist_start + BIN_HIST_SLM_SIZE;

    uint32_t slm_reorder = slm_reorder_start + local_tid * PROCESS_SIZE * sizeof(KeyT);
    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, PROCESS_SIZE> write_addr;
    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = PROCESS_SIZE * local_tid;

    #pragma unroll
    for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
        simd_mask<16> m = (io_offset+lane_id+s)<n;
        keys.template select<16, 1>(s) = merge(utils::gather(input, lane_id, io_offset + s), simd<KeyT, 16>(-1), m);
    }

    for (uint32_t stage=0; stage < STAGES; stage++) {
        bins = (keys >> (stage * RADIX_BITS)) & MASK;
        bin_offset = 0;
        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=1) {
            write_addr[s] = bin_offset[bins[s]];
            bin_offset[bins[s]] += 1;
        }

        /*
        first write to slm,
        then do column scan by group, each thread to 32c*8r,
        then last row do exclusive scan as incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        {
            barrier();
            constexpr uint32_t HIST_STRIDE = sizeof(hist_t)*BIN_COUNT;
            #pragma unroll
            for (uint32_t s = 0; s<BIN_COUNT; s+=128) {
                lsc_slm_block_store<uint32_t, 64>(slm_bin_hist_this_thread + s*sizeof(hist_t), bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE) {
                constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;
                constexpr uint32_t BIN_WIDTH_UD = BIN_WIDTH * sizeof(hist_t)/sizeof(uint32_t);
                uint32_t slm_bin_hist_summary_offset = slm_bin_hist_start + local_tid * BIN_WIDTH * sizeof(hist_t);
                simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
                simd<uint32_t, BIN_WIDTH_UD> tmp;

                thread_grf_hist_summary.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s<THREAD_PER_TG-1; s++) {
                    tmp = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                    thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset, thread_grf_hist_summary.template bit_cast_view<uint32_t>());
                    slm_bin_hist_summary_offset += HIST_STRIDE;
                }
                tmp = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                thread_grf_hist_summary = utils::scan<hist_t, hist_t>(thread_grf_hist_summary);
                lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset, thread_grf_hist_summary.template bit_cast_view<uint32_t>());
            }
            barrier();
            if (local_tid == 0) {
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT+1> grf_hist_summary_scan;
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=128) {
                    grf_hist_summary.template select<128, 1>(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (THREAD_PER_TG-1) * HIST_STRIDE + s*sizeof(hist_t));
                }
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<32, 1>(1) = grf_hist_summary.template select<32, 1>(0);
                #pragma unroll
                for (uint32_t i = 32; i<BIN_COUNT; i+=32) {
                    grf_hist_summary_scan.template select<32, 1>(i+1) = grf_hist_summary.template select<32, 1>(i) + grf_hist_summary_scan[i];
                }
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=128) {
                    lsc_slm_block_store<uint32_t, 64>(slm_incoming_offset + s * sizeof(hist_t), grf_hist_summary_scan.template select<128, 1>(s).template bit_cast_view<uint32_t>());
                }
            }
            barrier();
            {
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=128) {
                    bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_incoming_offset + s*sizeof(hist_t));
                }
                if (local_tid>0) {
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=128) {
                        simd<hist_t, 128> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (local_tid-1)*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<128, 1>(s) += group_local_sum;
                    }
                }
            }
            barrier();
        }

        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
            simd<uint16_t, 16> bins_uw = bins.template select<16, 1>(s);
            write_addr.template select<16, 1>(s) += bin_offset.template iselect(bins_uw);
        }

        if (stage != STAGES - 1) {
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                slm_scatter<KeyT, 16>(
                    write_addr.template select<16, 1>(s)*sizeof(KeyT) + slm_reorder_start,
                    keys.template select<16, 1>(s));
            }
            barrier();
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=64){
                keys.template select<64, 1>(s) = lsc_slm_block_load<KeyT, 64>(slm_reorder+s*sizeof(KeyT));
            }
        }
    }
    #pragma unroll
    for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
        utils::scatter<KeyT, 16>(input, write_addr.template select<16, 1>(s) * sizeof(KeyT),
                                 keys.template select<16, 1>(s),
                                 (local_tid * PROCESS_SIZE + lane_id + s) < n);
    }
}

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_one_wg;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t PROCESS_SIZE, typename _KernelName>
struct __radix_sort_one_wg_submitter;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t PROCESS_SIZE, typename... _Name>
struct __radix_sort_one_wg_submitter<KeyT, RADIX_BITS, PROCESS_SIZE,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, ::std::size_t __n, ::std::uint32_t __tg_count) const
    {
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        sycl::nd_range<1> __nd_range{__tg_count, __tg_count};

        return __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __data = __rng.data();
            __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        one_wg_kernel<KeyT,  decltype(__data), RADIX_BITS, PROCESS_SIZE> (
                            __nd_item, __n, __tg_count, __data);
                    });
        });
    }
};

template <typename _ExecutionPolicy, typename KeyT, typename _Range, ::std::uint32_t RADIX_BITS>
void one_wg(_ExecutionPolicy&& __exec, _Range&& __rng, ::std::size_t __n) {
    using namespace sycl;
    using namespace __ESIMD_NS;

    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t MAX_TG_COUNT = 64;
    constexpr uint32_t MIN_TG_COUNT = 8;
    uint32_t PROCESS_SIZE = 64;

    if (__n < MIN_TG_COUNT*64)
    {
        PROCESS_SIZE = 64;
    }
    else if (__n < MIN_TG_COUNT*128)
    {
        PROCESS_SIZE = 128;
    }
    else
    {
        PROCESS_SIZE = 256;
    }

    uint32_t TG_COUNT = oneapi::dpl::__internal::__dpl_ceiling_div(__n, PROCESS_SIZE);
    TG_COUNT = std::max(TG_COUNT, MIN_TG_COUNT);

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _EsimRadixSortKernel =
    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__esimd_radix_sort_one_wg<_CustomName>>;

    sycl::event __e;
    if (PROCESS_SIZE == 64)
    {
        __e = __radix_sort_one_wg_submitter<KeyT, RADIX_BITS, 64, _EsimRadixSortKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __n, TG_COUNT);
    }

    else if (PROCESS_SIZE == 128)
    {
        __e = __radix_sort_one_wg_submitter<KeyT, RADIX_BITS, 128, _EsimRadixSortKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __n, TG_COUNT);
    }
    else
    {
        __e = __radix_sort_one_wg_submitter<KeyT, RADIX_BITS, 256, _EsimRadixSortKernel>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __n, TG_COUNT);
    }
    __e.wait();
}

} // oneapi::dpl::experimental::esimd::impl

#endif // _ONEDPL_esimd_radix_sort_one_wg_H
