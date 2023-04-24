// -*- C++ -*-
//===-- esimd_radix_sort_onesweep.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H

#include <ext/intel/esimd.hpp>
#include <ext/intel/experimental/esimd/memory.hpp>

#include "../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include <cstdint>

namespace oneapi::dpl::experimental::esimd::impl
{
template <typename T>
struct SyclFreeOnDestroy
{
    sycl::queue __queue;
    T* __buffer = nullptr;

    SyclFreeOnDestroy(sycl::queue queue, T* buffer) : __queue(queue), __buffer(buffer)
    {
    }
    ~SyclFreeOnDestroy()
    {
        if (nullptr != __buffer)
        {
            sycl::free(__buffer, __queue);
        }
    }
};

template <typename T1, typename T2>
constexpr auto
div_up(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

template <typename T, int N>
inline std::enable_if_t<(N > 16) && (N % 16 == 0), __ESIMD_NS::simd<T, N>>
create_simd(T initial, T step)
{
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    simd<T, N> ret;
    ret.template select<16, 1>(0) = simd<T, 16>(0, 1) * step + initial;
    fence<fence_mask::sw_barrier>();
#pragma unroll
    for (int pos = 16; pos < N; pos += 16)
    {
        ret.template select<16, 1>(pos) = ret.template select<16, 1>(0) + pos * step;
    }
    return ret;
}


template <typename KeyT, typename InputT, uint32_t RADIX_BITS, uint32_t TG_COUNT, uint32_t THREAD_PER_TG, bool IsAscending>
void
global_histogram(sycl::nd_item<1> idx, size_t __n, const InputT& input, uint32_t *p_global_offset, uint32_t *p_sync_buffer)
{
    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;

    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using device_addr_t = uint32_t;

    slm_init(16384);
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);
    constexpr uint32_t PROCESS_SIZE = 128;
    constexpr uint32_t addr_step = TG_COUNT * THREAD_PER_TG * PROCESS_SIZE;

    uint32_t local_tid = idx.get_local_linear_id();
    uint32_t tid = idx.get_global_linear_id();

    constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;
    if (tid < SYNC_SEGMENT_COUNT)
    {
        simd<uint32_t, SYNC_SEGMENT_SIZE_DW> sync_init = 0;
        sync_init.copy_to(p_sync_buffer + SYNC_SEGMENT_SIZE_DW * tid);
    }

    if ((tid - local_tid) * PROCESS_SIZE > __n)
    {
        //no work for this tg;
        return;
    }

    // onesweep fill 0
    {
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        slm_block_store<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t), 0);
    }
    barrier();


    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<global_hist_t, BINCOUNT * STAGES> state_hist_grf(0);
    constexpr bin_t MASK = BINCOUNT - 1;

    device_addr_t read_addr;
    for (read_addr = tid * PROCESS_SIZE; read_addr < __n; read_addr += addr_step)
    {
        if (read_addr+PROCESS_SIZE < __n)
        {
            utils::copy_from(input, read_addr, keys);
        }
        else
        {
            simd<uint32_t, 16> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=16)
            {
                simd_mask<16> m = (s+lane_id)<(__n-read_addr);

                // simd<KeyT, 16> source = lsc_gather<KeyT, 1, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(input+read_addr+s, lane_id*sizeof(KeyT), m);
                sycl::ext::intel::esimd::simd offset((read_addr + s + lane_id)*sizeof(KeyT));
                simd<KeyT, 16> source = lsc_gather<KeyT, 1, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(input, offset, m);

                keys.template select<16, 1>(s) = merge(source, simd<KeyT, 16>(-1), m);
            }
        }
        #pragma unroll
        for (uint32_t stage = 0; stage < STAGES; stage++)
        {
            // bins = (keys >> (stage * RADIX_BITS)) & MASK;
            bins = oneapi::dpl::experimental::esimd::impl::utils::__get_bucket<MASK>(
                oneapi::dpl::experimental::esimd::impl::utils::__order_preserving_cast<IsAscending>(keys), stage * RADIX_BITS);

            #pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s++)
            {
                state_hist_grf[stage * BINCOUNT + bins[s]]++;// 256K * 4 * 1.25 = 1310720 instr for grf indirect addr
            }
        }
    }

    //atomic add to the state counter in slm
    #pragma unroll
    for (uint32_t s = 0; s < BINCOUNT * STAGES; s+=16)
    {
        simd<uint32_t, 16> offset(0, sizeof(global_hist_t));
        lsc_slm_atomic_update<atomic_op::add, global_hist_t, 16>(s*sizeof(global_hist_t)+offset, state_hist_grf.template select<16, 1>(s), 1);
    }

    barrier();

    {
        // bin count 256, 4 stages, 1K uint32_t, by 64 threads, happen to by 16-wide each thread. will not work for other config.
        constexpr uint32_t BUFFER_SIZE = STAGES * BINCOUNT;
        constexpr uint32_t THREAD_SIZE = BUFFER_SIZE / THREAD_PER_TG;
        simd<global_hist_t, THREAD_SIZE> group_hist = slm_block_load<global_hist_t, THREAD_SIZE>(local_tid*THREAD_SIZE*sizeof(global_hist_t));
        simd<uint32_t, THREAD_SIZE> offset(0, 4);
        lsc_atomic_update<atomic_op::add>(p_global_offset + local_tid*THREAD_SIZE, offset, group_hist, simd_mask<THREAD_SIZE>(1));
    }
}

inline
void
global_wait(uint32_t *psync, uint32_t sync_id, uint32_t count, uint32_t gid, uint32_t tid)
{
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    //assume initial is 1, do inc, then repeat load until count is met, then the first one atomic reduce by count to reset to 1, do not use store to 1, because second round might started.
    psync += sync_id;
    uint32_t current = -1;
    uint32_t try_count = 0;
    while (current != count)
    {
        // current=lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, 0, 1)[0];
        current = lsc_atomic_update<atomic_op::load, uint32_t, 1>(psync, simd<uint32_t, 1>(0), 1)[0];
        if (try_count++ > 5120)
            break;
    }
}

// onesweep
// +->  __radix_sort_onesweep_submitter::operator()
//          +-> submit()
//                  +-> parallel_for()
//                          +-> radix_sort_onesweep_slm_reorder_kernel (onesweep_kernel)
//      template <typename KeyT, typename InputT, typename OutputT, uint32_t RADIX_BITS, uint32_t THREAD_PER_TG,
//                uint32_t PROCESS_SIZE, bool IsAscending>
//      void
//      onesweep_kernel(sycl::nd_item<1> idx, uint32_t __n, uint32_t stage, const InputT& input, const OutputT& __output,
//                      uint8_t* p_global_buffer);

////////////////////////////////////////////////////////////////////////////////
// KSATODO required to copy implementation into onesweep_kernel
//using radix_sort_key_t = uint32_t;  // -> typename KeyT

template <typename KeyT, typename InputT, typename OutputT,
          uint32_t RADIX_BITS, uint32_t SG_PER_WG, uint32_t PROCESS_SIZE,
          bool IsAscending>
class radix_sort_onesweep_slm_reorder_kernel
{
    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    static constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    static constexpr uint32_t NBITS = sizeof(KeyT) * 8;
    static constexpr uint32_t STAGES = div_up(NBITS, RADIX_BITS);
    static constexpr bin_t MASK = BIN_COUNT - 1;
    static constexpr uint32_t REORDER_SLM_SIZE = PROCESS_SIZE * sizeof(KeyT) * SG_PER_WG;             // reorder buffer
    static constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * SG_PER_WG;             // bin hist working buffer
    static constexpr uint32_t SUBGROUP_LOOKUP_SIZE = BIN_COUNT * sizeof(hist_t) * SG_PER_WG;          // group offset lookup
    static constexpr uint32_t GLOBAL_LOOKUP_SIZE = BIN_COUNT * sizeof(global_hist_t);                 // global fix look up
    static constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT + 1) * sizeof(hist_t);            // incoming offset buffer

    //slm allocation:
    // first stage, do subgroup ranks, need SG_PER_WG*BIN_COUNT*sizeof(hist_t)
    // then do rank roll up in workgroup, need SG_PER_WG*BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(global_hist_t)
    // after all these is done, update ranks to workgroup ranks, need SUBGROUP_LOOKUP_SIZE
    // then shuffle keys to workgroup order in SLM, need PROCESS_SIZE * sizeof(KeyT) * SG_PER_WG
    // then read reordered slm and look up global fix, need GLOBAL_LOOKUP_SIZE on top
    static constexpr uint32_t slm_bin_hist_start = 0;
    static constexpr uint32_t slm_lookup_workgroup = 0;
    static constexpr uint32_t slm_reorder_start = 0;
    static constexpr uint32_t slm_lookup_global = slm_reorder_start + REORDER_SLM_SIZE;

    ::std::size_t n = 0;
    uint32_t stage = 0;
    InputT   p_input;                       // instance of sycl::accessor or pointer to input data
    OutputT  p_output = nullptr;            // pointer to output data
    uint8_t* p_global_buffer = nullptr;

    struct dynamic_job_queue_t
    {
        uint32_t* queue = nullptr;
        uint32_t slm = 0;
        dynamic_job_queue_t(uint32_t* queue) : queue(queue) {}
        dynamic_job_queue_t(uint32_t* queue, uint32_t slm) : queue(queue), slm(slm) {}

        template <typename TGroup>
        inline uint32_t
        get_job_id(TGroup g) const
        {
            using namespace __ESIMD_NS;
            using namespace __ESIMD_ENS;
            simd<uint32_t, 1> job_id;
            if (g.get_local_linear_id() == 0)
            {
                job_id = lsc_atomic_update<atomic_op::inc, uint32_t, 1>(queue, 0, 1);
                lsc_slm_scatter<uint32_t, 1, lsc_data_size::default_size, 1>(slm, job_id);
            }

            // KSATOD what is the barrier() function? Where it's implemented?
            barrier();
            if (g.get_local_linear_id() != 0)
            {
                job_id = lsc_slm_gather<uint32_t, 1, lsc_data_size::default_size, 1>(slm);
            }
            return job_id[0];
        }
    };
    dynamic_job_queue_t job_queue;

    template <typename T>
    struct slm_lookup_t
    {
        uint32_t slm = 0;
        inline slm_lookup_t(uint32_t slm) : slm(slm) {}

        template <int TABLE_SIZE>
        inline void
        setup(__ESIMD_NS::simd<T, TABLE_SIZE> source) SYCL_ESIMD_FUNCTION
        {
            oneapi::dpl::experimental::esimd::impl::utils::BlockStore<T, TABLE_SIZE>(slm, source);
        }

        template <int N, typename TIndex>
        inline auto
        lookup(TIndex idx) SYCL_ESIMD_FUNCTION
        {
            return oneapi::dpl::experimental::esimd::impl::utils::VectorLoad<T, 1, N>(
                slm + __ESIMD_NS::simd<uint32_t, N>(idx) * sizeof(T));
        }

        template <int N, int TABLE_SIZE, typename TIndex>
        inline auto
        lookup(__ESIMD_NS::simd<T, TABLE_SIZE> source, TIndex idx) SYCL_ESIMD_FUNCTION
        {
            setup(source);
            return lookup<N>(idx);
        }
    }; // struct slm_lookup_t

  public:

    radix_sort_onesweep_slm_reorder_kernel(::std::size_t n, uint32_t stage, InputT p_input, OutputT p_output,
                                           uint8_t* p_global_buffer,
                                           uint32_t* p_job_queue);

    void
    operator()(sycl::nd_item<1> idx) const SYCL_ESIMD_KERNEL;

protected:

    template <uint32_t CHUNK_SIZE>
    inline void
    LoadKeys(uint32_t io_offset, __ESIMD_NS::simd<KeyT, PROCESS_SIZE>& keys, KeyT default_key) const
    {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        bool is_full_block = (io_offset + PROCESS_SIZE) <= n;
        if (is_full_block)
        {
            simd<uint32_t, CHUNK_SIZE> lane_id(0, 1);
#pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s += CHUNK_SIZE)
            {
                keys.template select<CHUNK_SIZE, 1>(s) =
                    lsc_gather(p_input + io_offset + s, lane_id * uint32_t(sizeof(KeyT)));
                //keys.template select<CHUNK_SIZE, 1>(s) = block_load<T, CHUNK_SIZE>(p_input+io_offset+s); //will fail strangely.
            }
        }
        else
        {
            simd<uint32_t, CHUNK_SIZE> lane_id(0, 1);
#pragma unroll
            for (uint32_t s = 0; s < PROCESS_SIZE; s += CHUNK_SIZE)
            {
                simd_mask<CHUNK_SIZE> m = (io_offset + lane_id + s) < n;
                keys.template select<CHUNK_SIZE, 1>(s) =
                    merge(lsc_gather(p_input + io_offset + s, lane_id * uint32_t(sizeof(KeyT))),
                          simd<KeyT, CHUNK_SIZE>(default_key), m);
            }
        }
    }

    inline void
    ResetBinCounters(uint32_t slm_bin_hist_this_thread) const
    {
        oneapi::dpl::experimental::esimd::impl::utils::BlockStore<hist_t, BIN_COUNT>(slm_bin_hist_this_thread, 0);
    }

    template <uint32_t BITS>
    inline __ESIMD_NS::simd<uint32_t, 32>
    match_bins(__ESIMD_NS::simd<uint32_t, 32> bins, uint32_t local_tid) const
    {
        //instruction count is 5*BITS, so 40 for 8 bits.
        //performance is about 2 u per bit for processing size 512 (will call this 16 times)
        // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
        // 8 bits is 12.8 us
        using namespace __ESIMD_NS;
        fence<fence_mask::sw_barrier>();
        simd<uint32_t, 32> matched_bins(0xffffffff);
#pragma unroll
        for (int i = 0; i < BITS; i++)
        {
            simd<uint32_t, 32> bit = (bins >> i) & 1; // and

            // KSATOD what is the merge template function? Where it's implemented?
            simd<uint32_t, 32> x = merge<uint32_t, 32>(0, -1, bit != 0); // sel
            // KSATOD what is the pack_mask function? Where it's implemented?
            uint32_t ones = pack_mask(bit != 0);      // mov
            matched_bins = matched_bins & (x ^ ones); // bfn
        }
        fence<fence_mask::sw_barrier>();
        return matched_bins;
    }

    inline auto
    RankSLM(__ESIMD_NS::simd<bin_t, PROCESS_SIZE> bins, uint32_t slm_counter_offset, uint32_t local_tid) const
    {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
        simd<uint32_t, PROCESS_SIZE> ranks;
        oneapi::dpl::experimental::esimd::impl::utils::BlockStore<hist_t, BIN_COUNT>(slm_counter_offset, 0);
        simd<uint32_t, 32> remove_right_lanes, lane_id(0, 1);
        remove_right_lanes = 0x7fffffff >> (31 - lane_id);
#pragma unroll
        for (uint32_t s = 0; s < PROCESS_SIZE; s += 32)
        {
            simd<uint32_t, 32> this_bins = bins.template select<32, 1>(s);
            simd<uint32_t, 32> matched_bins = match_bins<RADIX_BITS>(this_bins, local_tid); // 40 insts
            simd<uint32_t, 32> pre_rank, this_round_rank;
            pre_rank = oneapi::dpl::experimental::esimd::impl::utils::VectorLoad<hist_t, 1, 32>(
                slm_counter_offset + this_bins * sizeof(hist_t)); // 2 mad+load.slm
            auto matched_left_lanes = matched_bins & remove_right_lanes;
            this_round_rank = cbit(matched_left_lanes);
            auto this_round_count = cbit(matched_bins);
            auto rank_after = pre_rank + this_round_rank;
            auto is_leader = this_round_rank == this_round_count - 1;
            oneapi::dpl::experimental::esimd::impl::utils::VectorStore<hist_t, 1, 32>(
                slm_counter_offset + this_bins * sizeof(hist_t), rank_after + 1, is_leader);
            ranks.template select<32, 1>(s) = rank_after;
            // if (local_tid==0) {
            //     PRINTD(this_bins);
            //     PRINTX(matched_bins);
            //     PRINTD(rank);
            //     PRINTD(this_round_count);
            //     PRINTD(this_round_rank);
            //     PRINTD(rank_after);
            // }
            // break;
        }
        return ranks;
    }

    inline void
    UpdateGroupRank(uint32_t local_tid, uint32_t wg_id, __ESIMD_NS::simd<hist_t, BIN_COUNT>& subgroup_offset,
                    __ESIMD_NS::simd<global_hist_t, BIN_COUNT>& global_fix, global_hist_t* p_global_bin_prev_group,
                    global_hist_t* p_global_bin_this_group) const
    {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        /*
        first do column scan by group, each thread do 32c, 
        then last row do exclusive scan as group incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        constexpr uint32_t HIST_STRIDE = sizeof(hist_t) * BIN_COUNT;
        const uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * HIST_STRIDE;
        const uint32_t slm_bin_hist_group_incoming = slm_bin_hist_start + SG_PER_WG * HIST_STRIDE;
        const uint32_t slm_bin_hist_global_incoming = slm_bin_hist_group_incoming + HIST_STRIDE;
        constexpr uint32_t GLOBAL_ACCUMULATED = 0x40000000;
        constexpr uint32_t HIST_UPDATED = 0x80000000;
        constexpr uint32_t GLOBAL_OFFSET_MASK = 0x3fffffff;
        {
            // KSATOD what is the barrier() function? Where it's implemented?
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;

            simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE)
            {
                uint32_t slm_bin_hist_summary_offset = slm_bin_hist_start + local_tid * BIN_WIDTH * sizeof(hist_t);
                thread_grf_hist_summary = oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<hist_t, BIN_WIDTH>(
                    slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s < SG_PER_WG; s++, slm_bin_hist_summary_offset += HIST_STRIDE)
                {
                    thread_grf_hist_summary +=
                        oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<hist_t, BIN_WIDTH>(
                            slm_bin_hist_summary_offset);
                    oneapi::dpl::experimental::esimd::impl::utils::BlockStore(slm_bin_hist_summary_offset,
                                                                              thread_grf_hist_summary);
                }
                oneapi::dpl::experimental::esimd::impl::utils::BlockStore(
                    slm_bin_hist_group_incoming + local_tid * BIN_WIDTH * sizeof(hist_t),
                           scan<hist_t, hist_t>(thread_grf_hist_summary));
                if (wg_id != 0)
                    lsc_block_store<uint32_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached,
                                    cache_hint::write_back>(p_global_bin_this_group + local_tid * BIN_WIDTH,
                                                            thread_grf_hist_summary | HIST_UPDATED);
            }
            // KSATOD what is the barrier() function? Where it's implemented?
            barrier();
            if (local_tid == BIN_SUMMARY_GROUP_SIZE + 1)
            {
                // this thread to group scan
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT + 1> grf_hist_summary_scan;
                grf_hist_summary = oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<hist_t, BIN_COUNT>(
                    slm_bin_hist_group_incoming);
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<BIN_WIDTH, 1>(1) =
                    grf_hist_summary.template select<BIN_WIDTH, 1>(0);
#pragma unroll
                for (uint32_t i = BIN_WIDTH; i < BIN_COUNT; i += BIN_WIDTH)
                {
                    grf_hist_summary_scan.template select<BIN_WIDTH, 1>(i + 1) =
                        grf_hist_summary.template select<BIN_WIDTH, 1>(i) + grf_hist_summary_scan[i];
                }
                oneapi::dpl::experimental::esimd::impl::utils::BlockStore<hist_t, BIN_COUNT>(
                    slm_bin_hist_group_incoming,
                                              grf_hist_summary_scan.template select<BIN_COUNT, 1>());
            }
            else if (local_tid < BIN_SUMMARY_GROUP_SIZE)
            {
                // these threads to global sync and update
                simd<global_hist_t, BIN_WIDTH> prev_group_hist_sum(0), prev_group_hist;
                simd_mask<BIN_WIDTH> is_not_accumulated(1);
                do
                {
                    do
                    {
                        prev_group_hist =
                            lsc_block_load<global_hist_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached,
                                           cache_hint::cached>(p_global_bin_prev_group + local_tid * BIN_WIDTH);
                        fence<fence_mask::sw_barrier>();
                    } while (((prev_group_hist & HIST_UPDATED) == 0).any() && wg_id != 0);
                    prev_group_hist_sum.merge(prev_group_hist_sum + prev_group_hist, is_not_accumulated);
                    is_not_accumulated = (prev_group_hist_sum & GLOBAL_ACCUMULATED) == 0;
                    p_global_bin_prev_group -= BIN_COUNT;
                } while (is_not_accumulated.any() && wg_id != 0);
                prev_group_hist_sum &= GLOBAL_OFFSET_MASK;
                simd<global_hist_t, BIN_WIDTH> after_group_hist_sum = prev_group_hist_sum + thread_grf_hist_summary;
                lsc_block_store<uint32_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached,
                                cache_hint::write_back>(p_global_bin_this_group + local_tid * BIN_WIDTH,
                                                        after_group_hist_sum | HIST_UPDATED | GLOBAL_ACCUMULATED);

                lsc_slm_block_store<uint32_t, BIN_WIDTH>(
                    slm_bin_hist_global_incoming + local_tid * BIN_WIDTH * sizeof(global_hist_t), prev_group_hist_sum);
            }
            // KSATOD what is the barrier() function? Where it's implemented?
            barrier();
        }
        auto group_incoming =
            oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming);
        global_fix = oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<global_hist_t, BIN_COUNT>(
                         slm_bin_hist_global_incoming) -
                     group_incoming;
        if (local_tid > 0)
        {
            subgroup_offset =
                group_incoming + oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<hist_t, BIN_COUNT>(
                                     slm_bin_hist_start + (local_tid - 1) * HIST_STRIDE);
        }
        else
            subgroup_offset = group_incoming;
    }

};  // class radix_sort_onesweep_slm_reorder_kernel

//----------------------------------------------------------------------------//
template <typename KeyT, typename InputT, typename OutputT,
          uint32_t RADIX_BITS, uint32_t SG_PER_WG, uint32_t PROCESS_SIZE,
          bool IsAscending>
radix_sort_onesweep_slm_reorder_kernel<KeyT, InputT, OutputT, RADIX_BITS, SG_PER_WG, PROCESS_SIZE,
                                       IsAscending>::radix_sort_onesweep_slm_reorder_kernel(
    ::std::size_t n, uint32_t stage, InputT p_input, OutputT p_output, uint8_t* p_global_buffer, uint32_t* p_job_queue)
    : n(n), stage(stage), p_input(p_input), p_output(p_output), p_global_buffer(p_global_buffer),
      job_queue(p_job_queue + stage)
{
}

//----------------------------------------------------------------------------//
template <typename KeyT, typename InputT, typename OutputT,
          uint32_t RADIX_BITS, uint32_t SG_PER_WG, uint32_t PROCESS_SIZE,
          bool IsAscending>
void
radix_sort_onesweep_slm_reorder_kernel<KeyT, InputT, OutputT, RADIX_BITS, SG_PER_WG, PROCESS_SIZE,
                                       IsAscending>::operator()(sycl::nd_item<1> idx) const
    SYCL_ESIMD_KERNEL
{
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;
    slm_init(128 * 1024);

    uint32_t local_tid = idx.get_local_linear_id();
    // static job queue
    uint32_t wg_id = idx.get_group(0);
    uint32_t wg_size = idx.get_local_range(0);
    uint32_t wg_count = idx.get_group_range(0);

    static_assert(STAGES * RADIX_BITS == NBITS, "");

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  PROCESS_SIZE = 256, BIN_COUNT = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    // change slm to reuse

    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * BIN_COUNT * sizeof(hist_t);
    uint32_t slm_lookup_subgroup = slm_lookup_workgroup + local_tid * sizeof(hist_t) * BIN_COUNT;
    //uint32_t slm_lookup_global = slm_lookup_start;

    //global_hist_t * p_lookup_this = p_lookup+wg_

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<hist_t, PROCESS_SIZE> ranks;
    simd<KeyT, PROCESS_SIZE> keys;
    simd<bin_t, PROCESS_SIZE> bins;
    simd<device_addr_t, 16> lane_id(0, 1);

    device_addr_t io_offset = PROCESS_SIZE * (wg_id * wg_size + local_tid);
    constexpr KeyT default_key = -1;

    LoadKeys<16>(io_offset, keys, default_key);
    bins = (keys >> (stage * RADIX_BITS)) & MASK;
    if (n == 0)
    { // without this, result will be wrong.
        //PRINTD(keys);
        //PRINTD(bins);
    }

    ResetBinCounters(slm_bin_hist_this_thread);

    fence<fence_mask::local_barrier>();

    ranks = RankSLM(bins, slm_bin_hist_this_thread, local_tid);

    // KSATOD what is the barrier() function? Where it's implemented?
    barrier();

    simd<hist_t, BIN_COUNT> subgroup_offset;
    simd<global_hist_t, BIN_COUNT> global_fix;

    //sync buffer is like below:
    // uint32_t hist_stages[STAGES][BIN_COUNT];
    // uint32_t sync_buffer[STAGES][wg_count][BIN_COUNT];
    global_hist_t* p_global_bin_start_buffer_allstages = reinterpret_cast<global_hist_t*>(p_global_buffer);
    global_hist_t* p_global_bin_start_buffer =
        p_global_bin_start_buffer_allstages + BIN_COUNT * STAGES + BIN_COUNT * wg_count * stage;

    global_hist_t* p_global_bin_this_group = p_global_bin_start_buffer + BIN_COUNT * wg_id;
    global_hist_t* p_global_bin_prev_group = p_global_bin_start_buffer + BIN_COUNT * (wg_id - 1);
    p_global_bin_prev_group = (0 == wg_id) ? (p_global_bin_start_buffer_allstages + BIN_COUNT * stage)
                                           : (p_global_bin_this_group - BIN_COUNT);

    UpdateGroupRank(local_tid, wg_id, subgroup_offset, global_fix, p_global_bin_prev_group, p_global_bin_this_group);

    // KSATOD what is the barrier() function? Where it's implemented?
    barrier();
    {
        //bins = (keys >> (stage * RADIX_BITS)) & MASK;

        slm_lookup_t<hist_t> subgroup_lookup(slm_lookup_subgroup);
        simd<hist_t, PROCESS_SIZE> wg_offset =
            ranks + subgroup_lookup.template lookup<PROCESS_SIZE>(subgroup_offset, bins);
        barrier();

        oneapi::dpl::experimental::esimd::impl::utils::VectorStore<KeyT, 1, PROCESS_SIZE>(
            simd<uint32_t, PROCESS_SIZE>(wg_offset) * sizeof(KeyT) + slm_reorder_start,
                                           keys);
    }
    barrier();
    slm_lookup_t<hist_t> l(slm_lookup_global);
    if (local_tid == 0)
    {
        l.template setup(global_fix);
    }
    barrier();
    {
        keys = oneapi::dpl::experimental::esimd::impl::utils::BlockLoad<KeyT, PROCESS_SIZE>(
            slm_reorder_start + local_tid * PROCESS_SIZE * sizeof(KeyT));
        // original impl :
        //      bins = (keys >> (stage * RADIX_BITS)) & MASK;
        // our current impl :
        bins = oneapi::dpl::experimental::esimd::impl::utils::__get_bucket<MASK>(
            oneapi::dpl::experimental::esimd::impl::utils::__order_preserving_cast<IsAscending>(keys),
            stage * RADIX_BITS);

        simd<hist_t, PROCESS_SIZE> group_offset = create_simd<hist_t, PROCESS_SIZE>(local_tid * PROCESS_SIZE, 1);

        simd<device_addr_t, PROCESS_SIZE> global_offset = group_offset + l.template lookup<PROCESS_SIZE>(bins);

        oneapi::dpl::experimental::esimd::impl::utils::VectorStore<KeyT, 1, PROCESS_SIZE>(
            p_output, global_offset * sizeof(KeyT), keys, global_offset < n);
    }
}

//----------------------------------------------------------------------------//


//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_onesweep_histogram;

template <typename... _Name>
class __esimd_radix_sort_onesweep_scan;

template <typename... _Name>
class __esimd_radix_sort_onesweep;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t HW_TG_COUNT, ::std::uint32_t THREAD_PER_TG,
          bool IsAscending, typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t HW_TG_COUNT, ::std::uint32_t THREAD_PER_TG,
          bool IsAscending, typename... _Name>
struct __radix_sort_onesweep_histogram_submitter<KeyT, RADIX_BITS, HW_TG_COUNT, THREAD_PER_TG, IsAscending,
                                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range, typename _GlobalOffsetData, typename _SyncData,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, const _GlobalOffsetData& __global_offset_data, const _SyncData& __sync_data,
               ::std::size_t __n) const
    {
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        sycl::nd_range<1> __nd_range(HW_TG_COUNT * THREAD_PER_TG, THREAD_PER_TG);
        return __exec.queue().submit([&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                        __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                            global_histogram<KeyT, decltype(__data), RADIX_BITS, HW_TG_COUNT, THREAD_PER_TG, IsAscending>(
                                __nd_item, __n, __data, __global_offset_data, __sync_data);
                        });
            });
    }
};

template <::std::uint32_t STAGES, ::std::uint32_t BINCOUNT, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t STAGES, ::std::uint32_t BINCOUNT, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<STAGES, BINCOUNT,
                                            oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _GlobalOffsetData,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, _GlobalOffsetData& __global_offset_data, ::std::size_t __n, const sycl::event __e) const
    {
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        sycl::nd_range<1> __nd_range(STAGES * BINCOUNT, BINCOUNT);
        return __exec.queue().submit([&](sycl::handler& __cgh)
            {
                __cgh.depends_on(__e);
                __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __nd_item)
                    {
                        uint32_t __offset = __nd_item.get_global_id(0);
                        auto __g = __nd_item.get_group();
                        uint32_t __count = __global_offset_data[__offset];
                        uint32_t __presum = sycl::exclusive_scan_over_group(__g, __count, sycl::plus<::std::uint32_t>());
                        __global_offset_data[__offset] = __presum;
                    });
            });
    }
};

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          bool IsAscending, typename _KernelName>
struct __radix_sort_onesweep_submitter;

// KSATODO It was renamed in Quolly's new version into radix_sort_onesweep_slm_reorder_kernel
// onesweep
// +->  __radix_sort_onesweep_submitter::operator()
//          +-> submit()
//                  +-> parallel_for()
//                          +-> radix_sort_onesweep_slm_reorder_kernel (onesweep_kernel)
template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          bool IsAscending, typename... _Name>
struct __radix_sort_onesweep_submitter<KeyT, RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE, IsAscending,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _ExecutionPolicy, typename _Range, typename _Output, typename _TmpData,
              oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
    sycl::event
    operator()(_ExecutionPolicy&& __exec, _Range&& __rng, _Output __output, _TmpData __tmp_data,
               ::std::uint32_t __sweep_tg_count, ::std::size_t __n, ::std::uint32_t* __p_job_queue,
               ::std::uint32_t __stage,
               const sycl::event& __e) const
    {
        _PRINT_INFO_IN_DEBUG_MODE(__exec);
        ::std::uint32_t __groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, PROCESS_SIZE * THREAD_PER_TG);
        sycl::nd_range<1> __nd_range(__sweep_tg_count * THREAD_PER_TG, THREAD_PER_TG);

        return __exec.queue().submit([&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __data = __rng.data();
                __cgh.depends_on(__e);

                // radix_sort_onesweep_slm_reorder_kernel<RADIX_BITS, THREAD_PER_TG, 256> K(
                //     n, stage, p_input, p_output, tmp_buffer, p_job_queue + stage);
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]]
                    {
                        // __data : sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::true_t>
                        // __output : unsigned int *&
                        // __tmp_data : unsigned char *

                        if (__stage % 2 == 0)
                        {
                            // onesweep_kernel<KeyT,
                            radix_sort_onesweep_slm_reorder_kernel<KeyT,             // typename KeyT
                                                                   decltype(__data), // typename InputT
                                                                   _Output,          // typename OutputT
                                                                   RADIX_BITS,       // uint32_t RADIX_BITS,
                                                                   THREAD_PER_TG,    // uint32_t SG_PER_WG
                                                                   PROCESS_SIZE,     // uint32_t PROCESS_SIZE
                                                                   IsAscending>      // bool IsAscending
                                kernelImpl(/* ::std::size_t n           */ __n,
                                           /* uint32_t stage            */ __stage,
                                           /* InputT* p_input           */ __data,
                                           /* OutputT* p_output,        */ __output,
                                           /* uint8_t * p_global_buffer */ __tmp_data,
                                           /* uint32_t * p_job_queue    */ __p_job_queue);      // + __stage - this part moved into constructor impl
                            kernelImpl(__nd_item);
                        }
                        else
                        {
                            radix_sort_onesweep_slm_reorder_kernel<KeyT,             // typename KeyT
                                                                   _Output,          // typename InputT
                                                                   decltype(__data), // typename OutputT
                                                                   RADIX_BITS,       // uint32_t RADIX_BITS,
                                                                   THREAD_PER_TG,    // uint32_t SG_PER_WG
                                                                   PROCESS_SIZE,     // uint32_t PROCESS_SIZE
                                                                   IsAscending>      // bool IsAscending
                                kernelImpl(/* ::std::size_t n           */ __n,
                                           /* uint32_t stage            */ __stage,
                                           /* InputT* p_input           */ __output,
                                           /* OutputT* p_output,        */ __data,
                                           /* uint8_t * p_global_buffer */ __tmp_data,
                                           /* uint32_t * p_job_queue    */ __p_job_queue);      // + __stage - this part moved into constructor impl
                                kernelImpl(__nd_item);
                        }
                    });
            });
    }
};

// onesweep
// +->  __radix_sort_onesweep_submitter::operator()
//          +-> submit()
//                  +-> parallel_for()
//                          +-> onesweep_kernel
template <typename _ExecutionPolicy, typename KeyT, typename _Range, ::std::uint32_t RADIX_BITS,
          bool IsAscending, ::std::uint32_t PROCESS_SIZE>
void onesweep(_ExecutionPolicy&& __exec, _Range&& __rng, ::std::size_t __n)
{
    static_assert(PROCESS_SIZE == 256 || PROCESS_SIZE == 384 || PROCESS_SIZE == 416,
                  "We are able to setup PROCESS_SIZE only as 256, 384 or 416");

    using namespace sycl;
    using namespace __ESIMD_NS;

    using _Policy = typename ::std::decay<_ExecutionPolicy>::type;
    using _CustomName = typename _Policy::kernel_name;
    using _EsimRadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_histogram<_CustomName>>;
    using _EsimRadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_scan<_CustomName>>;
    using _EsimRadixSort = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep<_CustomName>>;

    using global_hist_t = uint32_t;
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t HW_TG_COUNT = 64;
    constexpr uint32_t THREAD_PER_TG = 64;
    constexpr uint32_t SWEEP_PROCESSING_SIZE = PROCESS_SIZE; // uint32_t SWEEP_PROCESSING_SIZE = PROCESS_SIZE;
    const uint32_t sweep_tg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, THREAD_PER_TG*SWEEP_PROCESSING_SIZE);
    const uint32_t sweep_threads = sweep_tg_count * THREAD_PER_TG;
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);

    //assert((SWEEP_PROCESSING_SIZE == 256) || (SWEEP_PROCESSING_SIZE == 512) || (SWEEP_PROCESSING_SIZE == 1024) || (SWEEP_PROCESSING_SIZE == 1536));

    //constexpr uint32_t SYNC_SEGMENT_COUNT = 64;
    //constexpr uint32_t SYNC_SEGMENT_SIZE_DW = 128;
    //constexpr uint32_t SYNC_BUFFER_SIZE = SYNC_SEGMENT_COUNT * SYNC_SEGMENT_SIZE_DW * sizeof(uint32_t); //bytes
    const uint32_t SYNC_BUFFER_SIZE = sweep_tg_count * BINCOUNT * STAGES * sizeof(global_hist_t); //bytes
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES * sizeof(global_hist_t);
    const uint32_t LOOKUP_BUFFER_SIZE = sweep_tg_count * BINCOUNT * sizeof(global_hist_t);
    constexpr uint32_t JOB_QUEUE_SIZE = 256;
    const size_t temp_buffer_size = GLOBAL_OFFSET_SIZE + // global offset
                                    SYNC_BUFFER_SIZE +   // sync buffer
                                    JOB_QUEUE_SIZE +     // dynamic job queue
                                    LOOKUP_BUFFER_SIZE;  // L1 lookup buffer

    // KSATODO malloc_device may be unavailable on current device!
    uint8_t* tmp_buffer = sycl::malloc_device<uint8_t>(temp_buffer_size, __exec.queue());
    SyclFreeOnDestroy tmp_buffer_free(__exec.queue(), tmp_buffer);
    auto p_global_offset = reinterpret_cast<uint32_t*>(tmp_buffer);
    auto p_sync_buffer = reinterpret_cast<uint32_t*>(tmp_buffer + GLOBAL_OFFSET_SIZE);
    auto p_job_queue = reinterpret_cast<uint32_t*>(tmp_buffer + GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE);
    auto p_lookup = reinterpret_cast<uint32_t*>(tmp_buffer + GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE + JOB_QUEUE_SIZE);
    // KSATODO malloc_device may be unavailable on current device!
    auto __output = sycl::malloc_device<uint32_t>(__n, __exec.queue());
    SyclFreeOnDestroy __output_free(__exec.queue(), __output);

    //sycl::event __e_init = __exec.queue().memset(tmp_buffer, 0, temp_buffer_size);
    // KSATODO fixed error
    // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#table.members.handler.copy
    //     void memset(void* ptr, int value, size_t numBytes) - Fills numBytes bytes of memory beginning at address ptr with value.
    sycl::event __e_init = __exec.queue().memset(tmp_buffer, 0, /* size_t numBytes */ temp_buffer_size * sizeof(uint8_t));
    __e_init.wait();

    sycl::event __e = __radix_sort_onesweep_histogram_submitter<
        KeyT, RADIX_BITS, HW_TG_COUNT, THREAD_PER_TG, IsAscending, _EsimRadixSortHistogram>()(
            ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), p_global_offset, p_sync_buffer, __n);

    __e = __radix_sort_onesweep_scan_submitter<STAGES, BINCOUNT, _EsimRadixSortScan>()(
        ::std::forward<_ExecutionPolicy>(__exec), p_global_offset, __n, __e);

    for (::std::uint32_t __stage = 0; __stage < STAGES; __stage++)
    {
        // KSATODO required to pass p_job_queue + stage
        // KSATODO It was renamed in Quolly's new version into radix_sort_onesweep_slm_reorder_kernel

        // radix_sort_onesweep_slm_reorder_kernel<RADIX_BITS, THREAD_PER_TG, 256> K(
        //     n, stage, p_input, p_output, tmp_buffer, p_job_queue + stage);
        // cgh.parallel_for<class onesweep_256>(Range, K);
        // + RADIX_BITS                     (template param)
        // + THREAD_PER_TG                  (constexpr uint32_t)
        // + /*PROCESS_SIZE*/ 256, 384, 416
        //     -> SWEEP_PROCESSING_SIZE     (constexpr uint32_t)
        // + n -> __n                       (::std::size_t)
        // + stage -> __stage               (::std::uint32_t)
        // ? p_input
        // ? p_output
        // + tmp_buffer -> tmp_buffer       (uint8_t*)
        // + p_job_queue -> p_job_queue     (uint32_t*)
        // + stage -> __stage               (::std::uint32_t)

        __radix_sort_onesweep_submitter<KeyT,                       // typename KeyT
                                        RADIX_BITS,                 // ::std::uint32_t RADIX_BITS
                                        THREAD_PER_TG,              // ::std::uint32_t THREAD_PER_TG
                                        SWEEP_PROCESSING_SIZE,      // ::std::uint32_t PROCESS_SIZE
                                        IsAscending,                // bool IsAscending
                                        _EsimRadixSort> submitter;  // typename... _Name
        __e = submitter(
                ::std::forward<_ExecutionPolicy>(__exec),   // _ExecutionPolicy&& __exec
                ::std::forward<_Range>(__rng),              // _Range&& __rng
                __output,                                   // const _Output& __output
                tmp_buffer,                                 // const _TmpData& __tmp_data
                sweep_tg_count,                             // ::std::uint32_t __sweep_tg_count
                __n,                                        // ::std::size_t __n
                p_job_queue,                                // ::std::uint32_t* __p_job_queue
                __stage,                                    // ::std::uint32_t __stage
                __e);                                       // const sycl::event& __e
    }
    __e.wait();
}

} // oneapi::dpl::experimental::esimd::impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
