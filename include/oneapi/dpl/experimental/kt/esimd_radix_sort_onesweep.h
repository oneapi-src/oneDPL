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
#include "../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include <cstdint>

namespace oneapi::dpl::experimental::kt::esimd::impl
{

template <typename KeyT, typename InputT, uint32_t RADIX_BITS, uint32_t STAGES, uint32_t WORK_GROUPS,
          uint32_t WORK_GROUP_SIZE, bool IsAscending>
void global_histogram(sycl::nd_item<1> idx, size_t __n, const InputT& input, uint32_t *p_global_offset) {
    using namespace sycl;
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;

    slm_init(16384);

    constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    constexpr uint32_t DATA_PER_WORK_ITEM = 128;
    constexpr uint32_t DEVICE_WIDE_STEP = WORK_GROUPS * WORK_GROUP_SIZE * DATA_PER_WORK_ITEM;

    // Cap the number of histograms to reduce in SLM per input range read pass
    // due to excessive GRF usage for thread-local histograms
    constexpr uint32_t STAGES_PER_BLOCK = sizeof(KeyT) < 4? sizeof(KeyT) : 4;
    constexpr uint32_t STAGE_BLOCKS = oneapi::dpl::__internal::__dpl_ceiling_div(STAGES, STAGES_PER_BLOCK);

    constexpr uint32_t HIST_BUFFER_SIZE = STAGES * BIN_COUNT;
    constexpr uint32_t HIST_DATA_PER_WORK_ITEM = HIST_BUFFER_SIZE / WORK_GROUP_SIZE;

    simd<KeyT, DATA_PER_WORK_ITEM> keys;
    simd<bin_t, DATA_PER_WORK_ITEM> bins;

    uint32_t local_id = idx.get_local_linear_id();
    uint32_t global_id = idx.get_global_linear_id();

    // 0. Early exit for threads without work
    if ((global_id - local_id) * DATA_PER_WORK_ITEM > __n)
    {
        return;
    }

    // 1. Initialize group-local histograms in SLM
    utils::BlockStore<global_hist_t, HIST_DATA_PER_WORK_ITEM>(
        local_id * HIST_DATA_PER_WORK_ITEM * sizeof(global_hist_t), 0);
    barrier();

    #pragma unroll
    for(uint32_t stage_block = 0; stage_block < STAGE_BLOCKS; ++stage_block)
    {
        simd<global_hist_t, BIN_COUNT * STAGES_PER_BLOCK> state_hist_grf(0);
        uint32_t stage_block_start = stage_block * STAGES_PER_BLOCK;

        for (uint32_t wi_offset = global_id * DATA_PER_WORK_ITEM; wi_offset < __n; wi_offset += DEVICE_WIDE_STEP)
        {
            // 1. Read keys
            // TODO: avoid reading global memory twice when STAGE_BLOCKS > 1 increasing DATA_PER_WORK_ITEM
            if (wi_offset + DATA_PER_WORK_ITEM < __n)
            {
                utils::copy_from(input, wi_offset, keys);
            }
            else
            {
                constexpr uint8_t DATA_PER_STEP = 16;
                simd<uint32_t, DATA_PER_STEP> lane_offsets(0, 1);
                #pragma unroll
                for (uint32_t step_offset = 0; step_offset < DATA_PER_WORK_ITEM; step_offset += DATA_PER_STEP)
                {
                    simd<uint32_t, DATA_PER_STEP> byte_offsets = (lane_offsets + step_offset + wi_offset) * sizeof(KeyT);
                    simd_mask<DATA_PER_STEP> is_in_range = byte_offsets < __n * sizeof(KeyT);
                    simd<KeyT, DATA_PER_STEP> data = lsc_gather<KeyT>(input, byte_offsets, is_in_range);
                    simd<KeyT, DATA_PER_STEP> sort_identities = utils::__sort_identity<KeyT, IsAscending>();
                    keys.template select<DATA_PER_STEP, 1>(step_offset) = merge(data, sort_identities, is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
            #pragma unroll
            for (uint32_t stage_local = 0; stage_local < STAGES_PER_BLOCK; ++stage_local)
            {
                constexpr bin_t MASK = BIN_COUNT - 1;
                uint32_t stage_global = stage_block_start + stage_local;
                bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<IsAscending>(keys), stage_global * RADIX_BITS);
                #pragma unroll
                for (uint32_t i = 0; i < DATA_PER_WORK_ITEM; ++i)
                {
                    ++state_hist_grf[stage_local * BIN_COUNT + bins[i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
        constexpr uint8_t DATA_PER_STEP = 16;
        #pragma unroll
        for (uint32_t grf_offset = 0; grf_offset < BIN_COUNT * STAGES_PER_BLOCK; grf_offset += DATA_PER_STEP)
        {
            uint32_t slm_offset = stage_block_start * BIN_COUNT + grf_offset;
            simd<uint32_t, DATA_PER_STEP> slm_byte_offsets(slm_offset * sizeof(global_hist_t), sizeof(global_hist_t));
            lsc_slm_atomic_update<atomic_op::add, global_hist_t, DATA_PER_STEP>(
                slm_byte_offsets, state_hist_grf.template select<DATA_PER_STEP, 1>(grf_offset), 1);
        }
        barrier();
    }

    // 4. Reduce group-local historgrams from SLM into global histograms in global memory
    simd<global_hist_t, HIST_DATA_PER_WORK_ITEM> group_hist = utils::BlockLoad<global_hist_t, HIST_DATA_PER_WORK_ITEM>(
        local_id * HIST_DATA_PER_WORK_ITEM * sizeof(global_hist_t));
    simd<uint32_t, HIST_DATA_PER_WORK_ITEM> byte_offsets(0, sizeof(global_hist_t));
    lsc_atomic_update<atomic_op::add>(p_global_offset + local_id * HIST_DATA_PER_WORK_ITEM, byte_offsets, group_hist,
                                      simd_mask<HIST_DATA_PER_WORK_ITEM>(1));
}

template <uint32_t BITS>
inline __ESIMD_NS::simd<uint32_t, 32> match_bins(__ESIMD_NS::simd<uint32_t, 32> bins, uint32_t local_tid) {
    //instruction count is 5*BITS, so 40 for 8 bits.
    //performance is about 2 u per bit for processing size 512 (will call this 16 times)
    // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
    // 8 bits is 12.8 us
    using namespace __ESIMD_NS;
    fence<fence_mask::sw_barrier>();
    simd<uint32_t, 32> matched_bins(0xffffffff);
    #pragma unroll
    for (int i = 0; i<BITS; i++) {
        simd<uint32_t, 32> bit = (bins >> i) & 1;// and
        simd<uint32_t, 32> x = __ESIMD_NS::merge<uint32_t, 32>(0, -1, bit!=0); // sel
        uint32_t ones = pack_mask(bit!=0);// mov
        matched_bins = matched_bins & (x ^ ones); // bfn
    }
    fence<fence_mask::sw_barrier>();
    return matched_bins;
}

template <typename T>
struct slm_lookup_t {
    uint32_t slm;
    inline slm_lookup_t(uint32_t slm): slm(slm) {}

    template <int TABLE_SIZE>
    inline void setup (__ESIMD_NS::simd<T, TABLE_SIZE> source) SYCL_ESIMD_FUNCTION {
        utils::BlockStore<T, TABLE_SIZE>(slm, source);
    }

    template <int N, typename IDX>
    inline auto lookup (IDX idx) SYCL_ESIMD_FUNCTION {
        return utils::VectorLoad<T, 1, N>(slm+__ESIMD_NS::simd<uint32_t, N>(idx)*sizeof(T));
    }

    template <int N, int TABLE_SIZE, typename IDX>
    inline auto lookup (__ESIMD_NS::simd<T, TABLE_SIZE> source, IDX idx) SYCL_ESIMD_FUNCTION {
        setup(source);
        return lookup<N>(idx);
    }
};

template <typename KeyT, typename InputT, typename OutputT, uint32_t RADIX_BITS, uint32_t SG_PER_WG, uint32_t PROCESS_SIZE, bool IsAscending>
struct radix_sort_onesweep_slm_reorder_kernel {
    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    static constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
    static constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    static constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);
    static constexpr bin_t MASK = BIN_COUNT - 1;
    static constexpr uint32_t REORDER_SLM_SIZE = PROCESS_SIZE * sizeof(KeyT) * SG_PER_WG; // reorder buffer
    static constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * SG_PER_WG;             // bin hist working buffer
    static constexpr uint32_t SUBGROUP_LOOKUP_SIZE = BIN_COUNT * sizeof(hist_t) * SG_PER_WG;          // group offset lookup
    static constexpr uint32_t GLOBAL_LOOKUP_SIZE = BIN_COUNT * sizeof(global_hist_t);                 // global fix look up
    static constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+1)*sizeof(hist_t);                // incoming offset buffer

    //slm allocation:
    // first stage, do subgroup ranks, need SG_PER_WG*BIN_COUNT*sizeof(hist_t)
    // then do rank roll up in workgroup, need SG_PER_WG*BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(global_hist_t)
    // after all these is done, update ranks to workgroup ranks, need SUBGROUP_LOOKUP_SIZE
    // then shuffle keys to workgroup order in SLM, need PROCESS_SIZE * sizeof(KeyT) * SG_PER_WG
    // then read reordered slm and look up global fix, need GLOBAL_LOOKUP_SIZE on top

    uint32_t n;
    uint32_t stage;
    InputT input;
    OutputT output;
    uint8_t *p_global_buffer;

    radix_sort_onesweep_slm_reorder_kernel(uint32_t n, uint32_t stage, InputT input, OutputT output, uint8_t *p_global_buffer):
        n(n), stage(stage), input(input), output(output), p_global_buffer(p_global_buffer) {}

    template <uint32_t CHUNK_SIZE>
    inline void LoadKeys(uint32_t io_offset, __ESIMD_NS::simd<KeyT, PROCESS_SIZE> &keys, KeyT default_key) const {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        bool is_full_block = (io_offset+PROCESS_SIZE) < n;
        if (is_full_block) {
            simd<uint32_t, CHUNK_SIZE> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=CHUNK_SIZE) {
                sycl::ext::intel::esimd::simd offset((io_offset + s + lane_id) * sizeof(KeyT));
                keys.template select<CHUNK_SIZE, 1>(s) = lsc_gather<KeyT, 1, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(input, offset);
            }
        }
        else
        {
            simd<uint32_t, CHUNK_SIZE> lane_id(0, 1);
            #pragma unroll
            for (uint32_t s = 0; s<PROCESS_SIZE; s+=CHUNK_SIZE) {
                simd_mask<CHUNK_SIZE> m = (io_offset+lane_id+s)<n;

                sycl::ext::intel::esimd::simd offset((io_offset + s + lane_id) * sizeof(KeyT));
                keys.template select<CHUNK_SIZE, 1>(s) =
                    merge(
                        lsc_gather<KeyT, 1, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached, 16>(input, offset, m),
                        simd<KeyT, CHUNK_SIZE>(default_key),
                        m);
            }
        }
    }

    inline void ResetBinCounters(uint32_t slm_bin_hist_this_thread) const {
        utils::BlockStore<hist_t, BIN_COUNT>(slm_bin_hist_this_thread, 0);
    }

    inline auto RankSLM(__ESIMD_NS::simd<bin_t, PROCESS_SIZE> bins, uint32_t slm_counter_offset, uint32_t local_tid) const {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        constexpr uint32_t BIN_COUNT = 1 << RADIX_BITS;
        simd<uint32_t, PROCESS_SIZE> ranks;
        utils::BlockStore<hist_t, BIN_COUNT>(slm_counter_offset, 0);
        simd<uint32_t, 32> remove_right_lanes, lane_id(0, 1);
        remove_right_lanes = 0x7fffffff >> (31-lane_id);
        #pragma unroll
        for (uint32_t s = 0; s<PROCESS_SIZE; s+=32) {
            simd<uint32_t, 32> this_bins = bins.template select<32, 1>(s);
            simd<uint32_t, 32> matched_bins = match_bins<RADIX_BITS>(this_bins, local_tid); // 40 insts
            simd<uint32_t, 32> pre_rank, this_round_rank;
            pre_rank = utils::VectorLoad<hist_t, 1, 32>(slm_counter_offset + this_bins*sizeof(hist_t)); // 2 mad+load.slm
            auto matched_left_lanes = matched_bins & remove_right_lanes;
            this_round_rank = cbit(matched_left_lanes);
            auto this_round_count = cbit(matched_bins);
            auto rank_after = pre_rank + this_round_rank;
            auto is_leader = this_round_rank == this_round_count-1;
            utils::VectorStore<hist_t, 1, 32>(slm_counter_offset + this_bins*sizeof(hist_t), rank_after+1, is_leader);
            ranks.template select<32, 1>(s) = rank_after;
        }
        return ranks;
    }

    inline void UpdateGroupRank(
        uint32_t local_tid,
        uint32_t wg_id,
        __ESIMD_NS::simd<hist_t, BIN_COUNT> &subgroup_offset,
        __ESIMD_NS::simd<global_hist_t, BIN_COUNT> &global_fix,
        global_hist_t *p_global_bin_prev_group,
        global_hist_t *p_global_bin_this_group
        ) const {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        /*
        first do column scan by group, each thread do 32c,
        then last row do exclusive scan as group incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        constexpr uint32_t HIST_STRIDE = sizeof(hist_t)*BIN_COUNT;
        const uint32_t slm_bin_hist_this_thread = local_tid*HIST_STRIDE;
        const uint32_t slm_bin_hist_group_incoming = SG_PER_WG * HIST_STRIDE;
        const uint32_t slm_bin_hist_global_incoming = slm_bin_hist_group_incoming + HIST_STRIDE;
        constexpr uint32_t GLOBAL_ACCUMULATED = 0x40000000;
        constexpr uint32_t HIST_UPDATED = 0x80000000;
        constexpr uint32_t GLOBAL_OFFSET_MASK = 0x3fffffff;
        {
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;

            simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE) {
                uint32_t slm_bin_hist_summary_offset = local_tid * BIN_WIDTH * sizeof(hist_t);
                thread_grf_hist_summary = utils::BlockLoad<hist_t, BIN_WIDTH>(slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s<SG_PER_WG; s++, slm_bin_hist_summary_offset += HIST_STRIDE) {
                    thread_grf_hist_summary += utils::BlockLoad<hist_t, BIN_WIDTH>(slm_bin_hist_summary_offset);
                    utils::BlockStore(slm_bin_hist_summary_offset, thread_grf_hist_summary);
                }
                utils::BlockStore(slm_bin_hist_group_incoming + local_tid * BIN_WIDTH * sizeof(hist_t), utils::scan<hist_t, hist_t>(thread_grf_hist_summary));
                if (wg_id!=0)
                    lsc_block_store<uint32_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>(p_global_bin_this_group + local_tid*BIN_WIDTH, thread_grf_hist_summary | HIST_UPDATED);
            }
            barrier();
            if (local_tid == BIN_SUMMARY_GROUP_SIZE+1) {
                // this thread to group scan
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT+1> grf_hist_summary_scan;
                grf_hist_summary = utils::BlockLoad<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming);
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<BIN_WIDTH, 1>(1) = grf_hist_summary.template select<BIN_WIDTH, 1>(0);
                #pragma unroll
                for (uint32_t i = BIN_WIDTH; i<BIN_COUNT; i+=BIN_WIDTH) {
                    grf_hist_summary_scan.template select<BIN_WIDTH, 1>(i+1) = grf_hist_summary.template select<BIN_WIDTH, 1>(i) + grf_hist_summary_scan[i];
                }
                utils::BlockStore<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming, grf_hist_summary_scan.template select<BIN_COUNT, 1>());
            } else if (local_tid < BIN_SUMMARY_GROUP_SIZE) {
                // these threads to global sync and update
                simd<global_hist_t, BIN_WIDTH> prev_group_hist_sum(0), prev_group_hist;
                simd_mask<BIN_WIDTH> is_not_accumulated(1);
                do {
                    do {
                        prev_group_hist = lsc_block_load<global_hist_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached, cache_hint::cached>(p_global_bin_prev_group + local_tid*BIN_WIDTH);
                        fence<fence_mask::sw_barrier>();
                    } while (((prev_group_hist & HIST_UPDATED) == 0).any() && wg_id != 0);
                    prev_group_hist_sum.merge(prev_group_hist_sum + prev_group_hist, is_not_accumulated);
                    is_not_accumulated = (prev_group_hist_sum & GLOBAL_ACCUMULATED)==0;
                    p_global_bin_prev_group -= BIN_COUNT;
                } while (is_not_accumulated.any() && wg_id != 0);
                prev_group_hist_sum &= GLOBAL_OFFSET_MASK;
                simd<global_hist_t, BIN_WIDTH> after_group_hist_sum = prev_group_hist_sum + thread_grf_hist_summary;
                lsc_block_store<uint32_t, BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>(p_global_bin_this_group + local_tid*BIN_WIDTH, after_group_hist_sum | HIST_UPDATED | GLOBAL_ACCUMULATED);

                utils::BlockStore<uint32_t, BIN_WIDTH>(slm_bin_hist_global_incoming + local_tid * BIN_WIDTH * sizeof(global_hist_t), prev_group_hist_sum);
            }
            barrier();
        }
        auto group_incoming = utils::BlockLoad<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming);
        global_fix = utils::BlockLoad<global_hist_t, BIN_COUNT>(slm_bin_hist_global_incoming) - group_incoming;
        if (local_tid>0) {
            subgroup_offset = group_incoming + utils::BlockLoad<hist_t, BIN_COUNT>((local_tid-1)*HIST_STRIDE);
        }
        else
            subgroup_offset = group_incoming;
    }


    void operator() (sycl::nd_item<1> idx) const SYCL_ESIMD_KERNEL {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        slm_init(128*1024);

        uint32_t local_tid = idx.get_local_linear_id();
        uint32_t wg_id = idx.get_group(0);
        uint32_t wg_size = idx.get_local_range(0);
        uint32_t wg_count = idx.get_group_range(0);

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  PROCESS_SIZE = 256, BIN_COUNT = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change slm to reuse

        uint32_t slm_bin_hist_this_thread = local_tid * BIN_COUNT * sizeof(hist_t);
        uint32_t slm_lookup_subgroup = local_tid*sizeof(hist_t)*BIN_COUNT;

        simd<hist_t, BIN_COUNT> bin_offset;
        simd<hist_t, PROCESS_SIZE> ranks;
        simd<KeyT, PROCESS_SIZE> keys;
        simd<bin_t, PROCESS_SIZE> bins;
        simd<device_addr_t, 16> lane_id(0, 1);

        device_addr_t io_offset = PROCESS_SIZE * (wg_id*wg_size+local_tid);
        constexpr KeyT default_key = utils::__sort_identity<KeyT, IsAscending>();

        LoadKeys<16>(io_offset, keys, default_key);

        bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<IsAscending>(keys), stage * RADIX_BITS);

        ResetBinCounters(slm_bin_hist_this_thread);

        fence<fence_mask::local_barrier>();

        ranks = RankSLM(bins, slm_bin_hist_this_thread, local_tid);

        barrier();

        simd<hist_t, BIN_COUNT> subgroup_offset;
        simd<global_hist_t, BIN_COUNT> global_fix;

        global_hist_t *p_global_bin_start_buffer_allstages = reinterpret_cast<global_hist_t*>(p_global_buffer);
        global_hist_t *p_global_bin_start_buffer = p_global_bin_start_buffer_allstages + BIN_COUNT * STAGES + BIN_COUNT * wg_count * stage;

        global_hist_t *p_global_bin_this_group = p_global_bin_start_buffer + BIN_COUNT * wg_id;
        global_hist_t *p_global_bin_prev_group = p_global_bin_start_buffer + BIN_COUNT * (wg_id-1);
        p_global_bin_prev_group = (0 == wg_id) ? (p_global_bin_start_buffer_allstages + BIN_COUNT * stage) : (p_global_bin_this_group - BIN_COUNT);

        UpdateGroupRank(local_tid, wg_id, subgroup_offset, global_fix, p_global_bin_prev_group, p_global_bin_this_group);
        barrier();
        {
            bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<IsAscending>(keys), stage * RADIX_BITS);

            slm_lookup_t<hist_t> subgroup_lookup(slm_lookup_subgroup);
            simd<hist_t, PROCESS_SIZE> wg_offset = ranks + subgroup_lookup.template lookup<PROCESS_SIZE>(subgroup_offset, bins);
            barrier();

            utils::VectorStore<KeyT, 1, PROCESS_SIZE>(simd<uint32_t, PROCESS_SIZE>(wg_offset)*sizeof(KeyT), keys);
        }
        barrier();
        slm_lookup_t<global_hist_t> l(REORDER_SLM_SIZE);
        if (local_tid == 0) {
            l.template setup(global_fix);
        }
        barrier();
        {
            keys = utils::BlockLoad<KeyT, PROCESS_SIZE>(local_tid * PROCESS_SIZE * sizeof(KeyT));

            bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<IsAscending>(keys), stage * RADIX_BITS);

            simd<hist_t, PROCESS_SIZE> group_offset = utils::create_simd<hist_t, PROCESS_SIZE>(local_tid*PROCESS_SIZE, 1);

            simd<device_addr_t, PROCESS_SIZE> global_offset = group_offset + l.template lookup<PROCESS_SIZE>(bins);

            utils::VectorStore<KeyT, 1, PROCESS_SIZE>(output, global_offset * sizeof(KeyT), keys, global_offset<n);
        }
    }
};

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_onesweep_histogram;

template <typename... _Name>
class __esimd_radix_sort_onesweep_scan;

template <typename... _Name>
class __esimd_radix_sort_onesweep_even;

template <typename... _Name>
class __esimd_radix_sort_onesweep_odd;

template <typename... _Name>
class __esimd_radix_sort_onesweep_copyback;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t STAGES, ::std::uint32_t HW_TG_COUNT,
          ::std::uint32_t THREAD_PER_TG, bool IsAscending, typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t STAGES, ::std::uint32_t HW_TG_COUNT,
          ::std::uint32_t THREAD_PER_TG, bool IsAscending, typename... _Name>
struct __radix_sort_onesweep_histogram_submitter<KeyT, RADIX_BITS, STAGES, HW_TG_COUNT, THREAD_PER_TG, IsAscending,
                                                 oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _Range&& __rng, const _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(HW_TG_COUNT * THREAD_PER_TG, THREAD_PER_TG);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng);
            __cgh.depends_on(__e);
            auto __data = __rng.data();
            __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        global_histogram<KeyT, decltype(__data), RADIX_BITS, STAGES, HW_TG_COUNT, THREAD_PER_TG, IsAscending>(
                            __nd_item, __n, __data, __global_offset_data);
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
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _GlobalOffsetData& __global_offset_data, ::std::size_t __n, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(STAGES * BINCOUNT, BINCOUNT);
        return __q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) {
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

template <typename KeyT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          bool IsAscending, typename... _Name>
struct __radix_sort_onesweep_submitter<KeyT, RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE, IsAscending,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRange, typename _OutRange, typename _TmpData>
    sycl::event
    operator()(sycl::queue& __q, _InRange& __rng, _OutRange& __out_rng, const _TmpData& __tmp_data,
               ::std::uint32_t __sweep_tg_count, ::std::size_t __n, ::std::uint32_t __stage, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_tg_count * THREAD_PER_TG, THREAD_PER_TG);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng, __out_rng);
            auto __in_data = __rng.data();
            auto __out_data = __out_rng.data();
            __cgh.depends_on(__e);
            radix_sort_onesweep_slm_reorder_kernel<KeyT, decltype(__in_data), decltype(__out_data), RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE, IsAscending>
                K(__n, __stage, __in_data, __out_data, __tmp_data);
            __cgh.parallel_for<_Name...>(__nd_range, K);
        });
    }
};

template <typename KeyT, typename _KernelName>
struct __radix_sort_copyback_submitter;

template <typename KeyT, typename... _Name>
struct __radix_sort_copyback_submitter<KeyT, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _TmpRange, typename _OutRange>
    sycl::event
    operator()(sycl::queue& __q, _TmpRange& __tmp_rng, _OutRange& __out_rng, ::std::uint32_t __n, const sycl::event& __e) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __tmp_rng, __out_rng);
            // TODO: make sure that access is read_only for __tmp_data  and is write_only for __out_rng
            auto __tmp_data = __tmp_rng.data();
            auto __out_data = __out_rng.data();
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(sycl::range<1>{__n}, [=](sycl::item<1> __item) {
                auto __global_id = __item.get_linear_id();
                __out_data[__global_id] = __tmp_data[__global_id];
            });
        });
    }
};

template <typename _KernelName, typename KeyT, typename _Range, ::std::uint32_t RADIX_BITS,
          bool IsAscending, ::std::uint32_t PROCESS_SIZE>
void
onesweep(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
{
    using namespace sycl;
    using namespace __ESIMD_NS;

    using _EsimdRadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_histogram<_KernelName>>;
    using _EsimdRadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_scan<_KernelName>>;
    using _EsimdRadixSortSweepEven = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_even<_KernelName>>;
    using _EsimdRadixSortSweepOdd = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_odd<_KernelName>>;
    using _EsimdRadixSortCopyback = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_copyback<_KernelName>>;

    using global_hist_t = uint32_t;
    constexpr uint32_t BINCOUNT = 1 << RADIX_BITS;
    constexpr uint32_t HW_TG_COUNT = 64;
    constexpr uint32_t THREAD_PER_TG = 64;
    constexpr uint32_t SWEEP_PROCESSING_SIZE = PROCESS_SIZE;
    const uint32_t sweep_tg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, THREAD_PER_TG*SWEEP_PROCESSING_SIZE);
    constexpr uint32_t NBITS =  sizeof(KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, RADIX_BITS);

    // Memory for SYNC_BUFFER_SIZE is used by onesweep kernel implicitly
    // TODO: pass a pointer to the the memory allocated with SYNC_BUFFER_SIZE to onesweep kernel
    const uint32_t SYNC_BUFFER_SIZE = sweep_tg_count * BINCOUNT * STAGES * sizeof(global_hist_t); //bytes
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES * sizeof(global_hist_t);
    size_t temp_buffer_size = GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE;

    const size_t full_buffer_size_global_hist = temp_buffer_size * sizeof(uint8_t);
    const size_t full_buffer_size_output = __n * sizeof(KeyT);
    const size_t full_buffer_size = full_buffer_size_global_hist + full_buffer_size_output;

    uint8_t* p_temp_memory = sycl::malloc_device<uint8_t>(full_buffer_size, __q);

    uint8_t* p_globl_hist_buffer = p_temp_memory;
    auto p_global_offset = reinterpret_cast<uint32_t*>(p_globl_hist_buffer);

    // Memory for storing values sorted for an iteration
    auto p_output = reinterpret_cast<KeyT*>(p_temp_memory + full_buffer_size_global_hist);
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, decltype(p_output)>();
    auto __out_rng = __keep(p_output, p_output + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event event_chain = __q.memset(p_globl_hist_buffer, 0, temp_buffer_size);

    event_chain = __radix_sort_onesweep_histogram_submitter<
        KeyT, RADIX_BITS, STAGES, HW_TG_COUNT, THREAD_PER_TG, IsAscending, _EsimdRadixSortHistogram>()(
            __q, __rng, p_global_offset, __n, event_chain);

    event_chain = __radix_sort_onesweep_scan_submitter<STAGES, BINCOUNT, _EsimdRadixSortScan>()(
        __q, p_global_offset, __n, event_chain);

    for (uint32_t stage = 0; stage < STAGES; stage++) {
        if((stage % 2) == 0)
        {
            event_chain = __radix_sort_onesweep_submitter<
                    KeyT, RADIX_BITS, THREAD_PER_TG, SWEEP_PROCESSING_SIZE, IsAscending, _EsimdRadixSortSweepEven>()(
                        __q, __rng, __out_rng, p_globl_hist_buffer, sweep_tg_count, __n, stage, event_chain);
        }
        else
        {
            event_chain = __radix_sort_onesweep_submitter<
                    KeyT, RADIX_BITS, THREAD_PER_TG, SWEEP_PROCESSING_SIZE, IsAscending, _EsimdRadixSortSweepOdd>()(
                        __q, __out_rng, __rng, p_globl_hist_buffer, sweep_tg_count, __n, stage, event_chain);
        }
    }

    if constexpr (STAGES % 2 != 0)
    {
        event_chain = __radix_sort_copyback_submitter<KeyT, _EsimdRadixSortCopyback>()(
            __q, __out_rng, __rng, __n, event_chain);
    }
    event_chain.wait();

    sycl::free(p_temp_memory, __q);
}

} // oneapi::dpl::experimental::kt::esimd::impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
