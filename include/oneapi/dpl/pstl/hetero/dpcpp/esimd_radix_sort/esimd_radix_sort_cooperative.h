// -*- C++ -*-
//===-- esimd_radix_sort_cooperative.h --------------------------------===//
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

#ifndef _ONEDPL_esimd_radix_sort_cooperative_H
#define _ONEDPL_esimd_radix_sort_cooperative_H

#include <ext/intel/esimd.hpp>
#include "../sycl_defs.h"
#include "../execution_sycl_defs.h"
#include "../parallel_backend_sycl_utils.h"
#include "../utils_ranges_sycl.h"
#include <cstdint>

namespace oneapi::dpl::experimental::esimd::impl
{

void inline init_global_sync(uint32_t * psync, uint32_t tg_id) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    simd<uint32_t, 16> lane_id(0, 4);
    simd<uint32_t, 16> old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
    if (tg_id == 0) {
        if (!(old_value==1).all()) {
            lsc_atomic_update<atomic_op::store, uint32_t, 16>(psync, lane_id, 1, 1);
        }
    } else {
        uint32_t try_count = 0;
        while (!(old_value==1).all()) {
            old_value = lsc_atomic_update<atomic_op::load, uint32_t, 16>(psync, lane_id, 1);
            if (try_count++ > 10240) break;
        }
    }
}

void inline global_sync(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    // uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, 0, 1)[0];
    uint32_t prev = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
    uint32_t current;
    current = -1;
    uint32_t try_count = 0;
    while (current != count+1) {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120) break;
    }
}

template <typename KeyT, typename InputT, uint32_t RADIX_BITS, uint32_t THREAD_PER_TG, uint32_t PROCESS_SIZE>
void cooperative_kernel(sycl::nd_item<1> idx, size_t n, const InputT& input, uint32_t *p_global_buffer) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    uint32_t tg_id = idx.get_group(0);
    uint32_t tg_count = idx.get_group_range(0);

    uint32_t local_tid = idx.get_local_linear_id();
    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);
    constexpr uint32_t TG_PROCESS_SIZE = PROCESS_SIZE * THREAD_PER_TG;
    constexpr bin_t MASK = BIN_COUNT - 1;

    constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * THREAD_PER_TG;  // bin hist working buffer, 64K for DW hist
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t); // incoming offset buffer
    constexpr uint32_t GLOBAL_SCAN_SIZE = (BIN_COUNT+16)*sizeof(global_hist_t);

    slm_init( BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE + GLOBAL_SCAN_SIZE);
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_start = slm_bin_hist_start + BIN_HIST_SLM_SIZE;
    uint32_t slm_global_scan_start = slm_incoming_start + INCOMING_OFFSET_SLM_SIZE;


    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);

    uint32_t global_sync_buffer_size = 1024; //1K uint32_t for sync buffer
    uint32_t global_bin_start_buffer_size = BIN_COUNT * sizeof(global_hist_t)+16;
    uint32_t global_bin_hist_size = tg_count * BIN_COUNT * sizeof(global_hist_t);
    uint32_t *p_sync_buffer = p_global_buffer;
    uint32_t *p_global_bin_start_buffer = p_sync_buffer + global_sync_buffer_size;
    uint32_t *p_global_bin_hist = p_global_bin_start_buffer + global_bin_start_buffer_size;
    uint32_t *p_global_bin_hist_tg = p_global_bin_hist + tg_id * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t);

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, PROCESS_SIZE> write_addr;
    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = tg_id * TG_PROCESS_SIZE + PROCESS_SIZE * local_tid;

    constexpr uint32_t BIN_GROUPS = 8;
    constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
    constexpr uint32_t BIN_WIDTH_UD = BIN_COUNT / BIN_GROUPS * sizeof(hist_t) / sizeof(uint32_t);
    constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;

    if (local_tid == 0) init_global_sync(p_sync_buffer, tg_id);
    barrier();

    #pragma unroll
    for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
        simd_mask<16> m = (io_offset+lane_id+s)<n;
        simd<KeyT, 16> source = lsc_gather<KeyT, 1,
                // lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(input+io_offset+s, lane_id*uint32_t(sizeof(KeyT)), m);
                lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(input, sycl::ext::intel::esimd::simd<KeyT, 16>((lane_id + io_offset+s)*uint32_t(sizeof(KeyT))), m);
        keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
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
            // put local hist to slm
            #pragma unroll
            for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                lsc_slm_block_store<uint32_t, 64>(slm_bin_hist_this_thread + s*sizeof(hist_t), bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            // small group sum for local hist
            constexpr uint32_t BIN_GROUPS = 8;
            constexpr uint32_t THREAD_PER_BIN_GROUP = THREAD_PER_TG / BIN_GROUPS;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_GROUPS;
            constexpr uint32_t BIN_HEIGHT = THREAD_PER_TG / THREAD_PER_BIN_GROUP;
            constexpr uint32_t HIST_STRIDE = BIN_COUNT * sizeof(hist_t);
            uint32_t THREAD_GID = local_tid / THREAD_PER_BIN_GROUP;
            uint32_t THREAD_LTID = local_tid % THREAD_PER_BIN_GROUP;
            {
                uint32_t slm_bin_hist_ingroup_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) + THREAD_LTID * BIN_HEIGHT * HIST_STRIDE;
                utils::simd2d<hist_t, BIN_HEIGHT, BIN_WIDTH> thread_grf_hist;
                #pragma unroll
                for (uint32_t s = 0; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    thread_grf_hist.row(s) += thread_grf_hist.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<BIN_HEIGHT; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_ingroup_offset + s * HIST_STRIDE, thread_grf_hist.row(s).template bit_cast_view<uint32_t>());
                }
            }

            barrier();
            // thread group sum for groups
            if (THREAD_LTID == 0) {
                uint32_t slm_bin_hist_group_summary_offset = slm_bin_hist_start + THREAD_GID * BIN_WIDTH * sizeof(hist_t) +  (BIN_HEIGHT-1) * HIST_STRIDE;
                utils::simd2d<hist_t, THREAD_PER_BIN_GROUP, BIN_WIDTH> thread_grf_hist_summary;
                #pragma unroll
                for (uint32_t s = 0; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP; s++) {
                    thread_grf_hist_summary.row(s) += thread_grf_hist_summary.row(s-1);
                }
                #pragma unroll
                for (uint32_t s = 1; s<THREAD_PER_BIN_GROUP-1; s++) {
                    lsc_slm_block_store<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_group_summary_offset + s * BIN_HEIGHT * HIST_STRIDE, thread_grf_hist_summary.row(s).template bit_cast_view<uint32_t>());
                }
                simd<global_hist_t, BIN_WIDTH> group_hist_sum = thread_grf_hist_summary.row(THREAD_PER_BIN_GROUP-1);
                group_hist_sum.copy_to(p_global_bin_hist_tg + THREAD_GID * BIN_WIDTH * sizeof(global_hist_t)/sizeof(uint32_t));
            }
            lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
            barrier();

            // thread group scan -> change to coopeative group scan
            // first dump result to global hist buffer - done in previous part, last row is copy to global
            // global sync()
            // each tg calculate sum of previous tg
            // last tg then do exclusive scan and save to global scan buffer
            // global sync()
            // each tg read global scan buffer, add with prev tg, then put to slm incoming buffer
            if (local_tid == 0) {
                global_sync(p_sync_buffer, stage*4+0, tg_count, tg_id, local_tid);
            }
            barrier();
            {
                if (tg_id == tg_count-1) {
                    if (local_tid < 16)  {
                        //16 threads cooperative, each for BIN_COUNT/16 element.
                        uint32_t *p_global_hist_sum_per_thread = p_global_bin_hist + local_tid * (BIN_COUNT/16)* sizeof(global_hist_t)/sizeof(uint32_t);

                        simd<global_hist_t, BIN_COUNT/16> global_hist_sum(p_global_hist_sum_per_thread);
                        for (uint32_t i = 1; i<tg_count; i++) {
                            simd<global_hist_t, BIN_COUNT/16> global_hist(p_global_hist_sum_per_thread + i * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                            global_hist_sum += global_hist;
                            global_hist_sum.copy_to(p_global_hist_sum_per_thread + i * BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                        }
                        global_hist_sum = utils::scan<global_hist_t, global_hist_t>(global_hist_sum);
                        lsc_slm_block_store<global_hist_t, 16>(slm_global_scan_start + local_tid * (BIN_COUNT/16)* sizeof(global_hist_t), global_hist_sum);
                    }
                    barrier();//TODO: change to smaller named barrier
                    if (local_tid==0) {
                        simd<global_hist_t, BIN_COUNT/16> global_hist_sum;
                        global_hist_t prev(0);
                        #pragma unroll
                        for (uint32_t i = 0; i<BIN_COUNT; i+=16) {
                            global_hist_sum = prev + lsc_slm_block_load<global_hist_t, 16>(slm_global_scan_start + i * sizeof(global_hist_t));
                            prev = global_hist_sum[15];
                            global_hist_sum.copy_to(p_global_bin_start_buffer+1+i);
                        }
                        p_global_bin_start_buffer[0] = 0;
                    }
                    lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
                }
                if (local_tid == 0)  {
                    global_sync(p_sync_buffer, stage*4+1, tg_count, tg_id, local_tid);
                    {
                        simd<global_hist_t, BIN_COUNT> global_hist_start(p_global_bin_start_buffer);
                        if (tg_id != 0) {
                            global_hist_start += simd<global_hist_t, BIN_COUNT>(p_global_bin_hist_tg - BIN_COUNT * sizeof(global_hist_t)/sizeof(uint32_t));
                        }
                        #pragma unroll
                        for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                            lsc_slm_block_store<uint32_t, 64>(slm_incoming_start + s * sizeof(global_hist_t), global_hist_start.template select<64, 1>(s));
                        }
                    }
                }
            }
            barrier();
            {
                #pragma unroll
                for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                    bin_offset.template select<64, 1>(s).template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_incoming_start + s*sizeof(hist_t));
                }
                if (local_tid>0) {
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + (local_tid-1)*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_local_sum;
                    }
                }
                if ((local_tid > THREAD_PER_BIN_GROUP) && (local_tid % BIN_HEIGHT != 0)) {
                    uint32_t prev_cum_rowid = (local_tid-1) / BIN_HEIGHT * BIN_HEIGHT - 1;
                    #pragma unroll
                    for (uint32_t s = 0; s<BIN_COUNT; s+=64) {
                        simd<hist_t, 64> group_sum;
                        group_sum.template bit_cast_view<uint32_t>() = lsc_slm_block_load<uint32_t, 64>(slm_bin_hist_start + prev_cum_rowid*HIST_STRIDE + s*sizeof(hist_t));
                        bin_offset.template select<64, 1>(s) += group_sum;
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

        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
            lsc_scatter<KeyT, 1, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back, 16>(
                input,
                write_addr.template select<16, 1>(s)*sizeof(KeyT),
                keys.template select<16, 1>(s), write_addr.template select<16, 1>(s)<n);
        }
        if (stage != STAGES - 1) {
            lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::evict, lsc_scope::gpu>();
            barrier();
            if (local_tid == 0) {global_sync(p_sync_buffer, stage*4+2, tg_count, tg_id, local_tid);}
            barrier();
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16) {
                simd_mask<16> m = (io_offset+lane_id+s)<n;
                simd<KeyT, 16> reordered = lsc_gather<KeyT, 1,
                        lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached, 16>(
                            // input+io_offset+s, lane_id*uint32_t(sizeof(KeyT)), m);
                            input, sycl::ext::intel::esimd::simd<KeyT, 16>((lane_id + io_offset+s)*uint32_t(sizeof(KeyT))), m);
                keys.template select<16, 1>(s) = merge(reordered, simd<KeyT, 16>(-1), m);
            }
        }
    }
}

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_cooperative;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          typename _KernelName>
struct __radix_sort_cooperative_submitter;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          typename... _Name>
struct __radix_sort_cooperative_submitter<KeyT, RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE,
                                          oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range, typename _SyncData,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, ::std::size_t __n, const _SyncData& __sync_data) const
    {
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        ::std::uint32_t __groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, PROCESS_SIZE * THREAD_PER_TG);
        sycl::nd_range<1> __nd_range{THREAD_PER_TG * __groups, THREAD_PER_TG};

        return __exec.queue().submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            auto __data = __rng.data();
            __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        cooperative_kernel<KeyT, decltype(__data), RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE> (
                            __nd_item, __n, __data, __sync_data);
                    });
        });
    }
};

template <typename _ExecutionPolicy, typename KeyT, typename _Range, ::std::uint32_t RADIX_BITS>
void cooperative(_ExecutionPolicy&& __exec, _Range&& __rng, ::std::size_t __n) {
    using namespace sycl;
    using namespace __ESIMD_NS;

    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t THREAD_PER_TG = 64;
    uint32_t MAX_GROUPS = 56; //TODO: get from sycl api

    auto p_sync = static_cast<uint32_t*>(sycl::malloc_device(1024 + (MAX_GROUPS+2) * BIN_COUNT * sizeof(uint32_t), __exec.queue()));

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _EsimRadixSort = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_cooperative<_CustomName>>;

    sycl::event __e;
    if (__n <= 128 * THREAD_PER_TG * MAX_GROUPS)
    {
        __e = __radix_sort_cooperative_submitter<
            KeyT, RADIX_BITS, THREAD_PER_TG, /*PROCESS_SIZE*/ 128, _EsimRadixSort>()(
                ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __n, p_sync);
    }
    else if (__n <= 256 * THREAD_PER_TG * MAX_GROUPS)
    {
        __e = __radix_sort_cooperative_submitter<
            KeyT, RADIX_BITS, THREAD_PER_TG, /*PROCESS_SIZE*/ 256, _EsimRadixSort>()(
                ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __n, p_sync);
    }
    __e.wait();
    sycl::free(p_sync, __exec.queue());
}

} // oneapi::dpl::experimental::esimd::impl

#endif // _ONEDPL_esimd_radix_sort_cooperative_H
