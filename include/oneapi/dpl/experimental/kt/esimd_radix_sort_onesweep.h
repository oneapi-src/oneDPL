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
template <typename _KeyT, typename InputT, uint32_t _RadixBits, uint32_t STAGES, uint32_t WORK_GROUPS,
          uint32_t WORK_GROUP_SIZE, bool _IsAscending>
void
global_histogram(sycl::nd_item<1> idx, size_t __n, const InputT& input, uint32_t* p_global_offset)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

    using bin_t = uint16_t;
    using hist_t = uint32_t;
    using global_hist_t = uint32_t;

    slm_init(16384);

    constexpr uint32_t BIN_COUNT = 1 << _RadixBits;
    constexpr uint32_t DATA_PER_WORK_ITEM = 128;
    constexpr uint32_t DEVICE_WIDE_STEP = WORK_GROUPS * WORK_GROUP_SIZE * DATA_PER_WORK_ITEM;

    // Cap the number of histograms to reduce in SLM per input range read pass
    // due to excessive GRF usage for thread-local histograms
    constexpr uint32_t STAGES_PER_BLOCK = sizeof(_KeyT) < 4 ? sizeof(_KeyT) : 4;
    constexpr uint32_t STAGE_BLOCKS = oneapi::dpl::__internal::__dpl_ceiling_div(STAGES, STAGES_PER_BLOCK);

    constexpr uint32_t HIST_BUFFER_SIZE = STAGES * BIN_COUNT;
    constexpr uint32_t HIST_DATA_PER_WORK_ITEM = HIST_BUFFER_SIZE / WORK_GROUP_SIZE;

    simd<_KeyT, DATA_PER_WORK_ITEM> keys;
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
    for (uint32_t stage_block = 0; stage_block < STAGE_BLOCKS; ++stage_block)
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
                simd<uint32_t, DATA_PER_STEP> lane_offsets(0, 1);
#pragma unroll
                for (uint32_t step_offset = 0; step_offset < DATA_PER_WORK_ITEM; step_offset += DATA_PER_STEP)
                {
                    simd<uint32_t, DATA_PER_STEP> offsets = lane_offsets + step_offset + wi_offset;
                    simd_mask<DATA_PER_STEP> is_in_range = offsets < __n;
                    simd<_KeyT, DATA_PER_STEP> data = utils::gather<_KeyT, DATA_PER_STEP>(input, offsets, 0, is_in_range);
                    simd<_KeyT, DATA_PER_STEP> sort_identities = utils::__sort_identity<_KeyT, _IsAscending>();
                    keys.template select<DATA_PER_STEP, 1>(step_offset) = merge(data, sort_identities, is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
#pragma unroll
            for (uint32_t stage_local = 0; stage_local < STAGES_PER_BLOCK; ++stage_local)
            {
                constexpr bin_t MASK = BIN_COUNT - 1;
                uint32_t stage_global = stage_block_start + stage_local;
                bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<_IsAscending>(keys),
                                                 stage_global * _RadixBits);
#pragma unroll
                for (uint32_t i = 0; i < DATA_PER_WORK_ITEM; ++i)
                {
                    ++state_hist_grf[stage_local * BIN_COUNT + bins[i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
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
inline __dpl_esimd_ns::simd<uint32_t, 32>
match_bins(__dpl_esimd_ns::simd<uint32_t, 32> bins, uint32_t local_tid)
{
    //instruction count is 5*BITS, so 40 for 8 bits.
    //performance is about 2 u per bit for processing size 512 (will call this 16 times)
    // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
    // 8 bits is 12.8 us
    using namespace __dpl_esimd_ns;
    fence<fence_mask::sw_barrier>();
    simd<uint32_t, 32> matched_bins(0xffffffff);
#pragma unroll
    for (int i = 0; i < BITS; i++)
    {
        simd<uint32_t, 32> bit = (bins >> i) & 1;                                    // and
        simd<uint32_t, 32> x = __dpl_esimd_ns::merge<uint32_t, 32>(0, -1, bit != 0); // sel
        uint32_t ones = pack_mask(bit != 0);                                         // mov
        matched_bins = matched_bins & (x ^ ones);                                    // bfn
    }
    fence<fence_mask::sw_barrier>();
    return matched_bins;
}

template <typename T>
struct slm_lookup_t
{
    uint32_t slm;
    inline slm_lookup_t(uint32_t slm) : slm(slm) {}

    template <int TABLE_SIZE>
    inline void
    setup(__dpl_esimd_ns::simd<T, TABLE_SIZE> source) SYCL_ESIMD_FUNCTION
    {
        utils::BlockStore<T, TABLE_SIZE>(slm, source);
    }

    template <int N, typename IDX>
    inline auto
    lookup(IDX idx) SYCL_ESIMD_FUNCTION
    {
        return utils::VectorLoad<T, 1, N>(slm + __dpl_esimd_ns::simd<uint32_t, N>(idx) * sizeof(T));
    }

    template <int N, int TABLE_SIZE, typename IDX>
    inline auto
    lookup(__dpl_esimd_ns::simd<T, TABLE_SIZE> source, IDX idx) SYCL_ESIMD_FUNCTION
    {
        setup(source);
        return lookup<N>(idx);
    }
};

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename InputT, typename OutputT>
struct radix_sort_onesweep_slm_reorder_kernel
{
    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using global_hist_t = uint32_t;
    using device_addr_t = uint32_t;

    static constexpr uint32_t BIN_COUNT = 1 << _RadixBits;
    static constexpr uint32_t NBITS = sizeof(_KeyT) * 8;
    static constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, _RadixBits);
    static constexpr bin_t MASK = BIN_COUNT - 1;
    static constexpr uint32_t REORDER_SLM_SIZE = _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize;
    static constexpr uint32_t BIN_HIST_SLM_SIZE = BIN_COUNT * sizeof(hist_t) * _WorkGroupSize;
    static constexpr uint32_t SUBGROUP_LOOKUP_SIZE = BIN_COUNT * sizeof(hist_t) * _WorkGroupSize;
    static constexpr uint32_t GLOBAL_LOOKUP_SIZE = BIN_COUNT * sizeof(global_hist_t);
    static constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT+1)*sizeof(hist_t);
    static constexpr uint32_t OFFSETS_SLM_SIZE = BIN_HIST_SLM_SIZE + GLOBAL_LOOKUP_SIZE + INCOMING_OFFSET_SLM_SIZE;

    //slm allocation:
    // first stage, do subgroup ranks, need _WorkGroupSize*BIN_COUNT*sizeof(hist_t)
    // then do rank roll up in workgroup, need _WorkGroupSize*BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(hist_t) + BIN_COUNT*sizeof(global_hist_t)
    // after all these is done, update ranks to workgroup ranks, need SUBGROUP_LOOKUP_SIZE
    // then shuffle keys to workgroup order in SLM, need _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize
    // then read reordered slm and look up global fix, need GLOBAL_LOOKUP_SIZE on top

    const uint32_t n;
    const uint32_t stage;
    InputT input;
    OutputT output;
    uint8_t* p_global_buffer;

    radix_sort_onesweep_slm_reorder_kernel(uint32_t n, uint32_t stage, InputT input, OutputT output,
                                           uint8_t* p_global_buffer)
        : n(n), stage(stage), input(input), output(output), p_global_buffer(p_global_buffer)
    {
    }

    inline void
    LoadKeys(uint32_t io_offset, __dpl_esimd_ns::simd<_KeyT, _DataPerWorkItem>& keys, _KeyT default_key) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;
        bool is_full_block = (io_offset + _DataPerWorkItem) < n;
        if (is_full_block)
        {
            simd<uint32_t, DATA_PER_STEP> lane_id(0, 1);
#pragma unroll
            for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
            {
                __dpl_esimd_ns::simd offset = io_offset + s + lane_id;
                keys.template select<DATA_PER_STEP, 1>(s) = utils::gather<_KeyT, DATA_PER_STEP>(input, offset, 0);
            }
        }
        else
        {
            simd<uint32_t, DATA_PER_STEP> lane_id(0, 1);
#pragma unroll
            for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
            {
                __dpl_esimd_ns::simd offset = io_offset + s + lane_id;
                simd_mask<DATA_PER_STEP> m = offset < n;
                keys.template select<DATA_PER_STEP, 1>(s) = merge(utils::gather<_KeyT, DATA_PER_STEP>(input, offset, 0, m),
                                                                  simd<_KeyT, DATA_PER_STEP>(default_key), m);
            }
        }
    }

    inline void
    ResetBinCounters(uint32_t slm_bin_hist_this_thread) const
    {
        utils::BlockStore<hist_t, BIN_COUNT>(slm_bin_hist_this_thread, 0);
    }

    inline auto
    RankSLM(__dpl_esimd_ns::simd<bin_t, _DataPerWorkItem> bins, uint32_t slm_counter_offset, uint32_t local_tid) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        constexpr int BinsPerStep = 32;

        constexpr uint32_t BIN_COUNT = 1 << _RadixBits;
        simd<uint32_t, _DataPerWorkItem> ranks;
        utils::BlockStore<hist_t, BIN_COUNT>(slm_counter_offset, 0);
        simd<uint32_t, BinsPerStep> remove_right_lanes, lane_id(0, 1);
        remove_right_lanes = 0x7fffffff >> (BinsPerStep - 1 - lane_id);
#pragma unroll
        for (uint32_t s = 0; s < _DataPerWorkItem; s += BinsPerStep)
        {
            simd<uint32_t, BinsPerStep> this_bins = bins.template select<BinsPerStep, 1>(s);
            simd<uint32_t, BinsPerStep> matched_bins = match_bins<_RadixBits>(this_bins, local_tid); // 40 insts
            simd<uint32_t, BinsPerStep> pre_rank, this_round_rank;
            pre_rank = utils::VectorLoad<hist_t, 1, BinsPerStep>(slm_counter_offset +
                                                                 this_bins * sizeof(hist_t)); // 2 mad+load.slm
            auto matched_left_lanes = matched_bins & remove_right_lanes;
            this_round_rank = cbit(matched_left_lanes);
            auto this_round_count = cbit(matched_bins);
            auto rank_after = pre_rank + this_round_rank;
            auto is_leader = this_round_rank == this_round_count - 1;
            utils::VectorStore<hist_t, 1, BinsPerStep>(slm_counter_offset + this_bins * sizeof(hist_t), rank_after + 1,
                                                       is_leader);
            ranks.template select<BinsPerStep, 1>(s) = rank_after;
        }
        return ranks;
    }

    inline void
    UpdateGroupRank(uint32_t local_tid, uint32_t wg_id, __dpl_esimd_ns::simd<hist_t, BIN_COUNT>& subgroup_offset,
                    __dpl_esimd_ns::simd<global_hist_t, BIN_COUNT>& global_fix, global_hist_t* p_global_bin_prev_group,
                    global_hist_t* p_global_bin_this_group) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;
        /*
        first do column scan by group, each thread do 32c,
        then last row do exclusive scan as group incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        constexpr uint32_t HIST_STRIDE = sizeof(hist_t) * BIN_COUNT;
        const uint32_t slm_bin_hist_this_thread = local_tid * HIST_STRIDE;
        const uint32_t slm_bin_hist_group_incoming = _WorkGroupSize * HIST_STRIDE;
        const uint32_t slm_bin_hist_global_incoming = slm_bin_hist_group_incoming + HIST_STRIDE;
        constexpr uint32_t GLOBAL_ACCUMULATED = 0x40000000;
        constexpr uint32_t HIST_UPDATED = 0x80000000;
        constexpr uint32_t GLOBAL_OFFSET_MASK = 0x3fffffff;
        {
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;

            simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE)
            {
                uint32_t slm_bin_hist_summary_offset = local_tid * BIN_WIDTH * sizeof(hist_t);
                thread_grf_hist_summary = utils::BlockLoad<hist_t, BIN_WIDTH>(slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s < _WorkGroupSize; s++, slm_bin_hist_summary_offset += HIST_STRIDE)
                {
                    thread_grf_hist_summary += utils::BlockLoad<hist_t, BIN_WIDTH>(slm_bin_hist_summary_offset);
                    utils::BlockStore(slm_bin_hist_summary_offset, thread_grf_hist_summary);
                }
                utils::BlockStore(slm_bin_hist_group_incoming + local_tid * BIN_WIDTH * sizeof(hist_t),
                                  utils::scan<hist_t, hist_t>(thread_grf_hist_summary));
                if (wg_id != 0)
                    utils::BlockStore<uint32_t, BIN_WIDTH>(p_global_bin_this_group + local_tid * BIN_WIDTH,
                                                           thread_grf_hist_summary | HIST_UPDATED);
            }
            barrier();
            if (local_tid == BIN_SUMMARY_GROUP_SIZE + 1)
            {
                // this thread to group scan
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT + 1> grf_hist_summary_scan;
                grf_hist_summary = utils::BlockLoad<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming);
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<BIN_WIDTH, 1>(1) =
                    grf_hist_summary.template select<BIN_WIDTH, 1>(0);
#pragma unroll
                for (uint32_t i = BIN_WIDTH; i < BIN_COUNT; i += BIN_WIDTH)
                {
                    grf_hist_summary_scan.template select<BIN_WIDTH, 1>(i + 1) =
                        grf_hist_summary.template select<BIN_WIDTH, 1>(i) + grf_hist_summary_scan[i];
                }
                utils::BlockStore<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming,
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
                utils::BlockStore<uint32_t, BIN_WIDTH>(p_global_bin_this_group + local_tid * BIN_WIDTH,
                                                       after_group_hist_sum | HIST_UPDATED | GLOBAL_ACCUMULATED);

                utils::BlockStore<uint32_t, BIN_WIDTH>(
                    slm_bin_hist_global_incoming + local_tid * BIN_WIDTH * sizeof(global_hist_t), prev_group_hist_sum);
            }
            barrier();
        }
        auto group_incoming = utils::BlockLoad<hist_t, BIN_COUNT>(slm_bin_hist_group_incoming);
        global_fix = utils::BlockLoad<global_hist_t, BIN_COUNT>(slm_bin_hist_global_incoming) - group_incoming;
        if (local_tid > 0)
        {
            subgroup_offset = group_incoming + utils::BlockLoad<hist_t, BIN_COUNT>((local_tid - 1) * HIST_STRIDE);
        }
        else
            subgroup_offset = group_incoming;
    }

    void
    operator()(sycl::nd_item<1> idx) const SYCL_ESIMD_KERNEL
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        slm_init(128 * 1024);

        uint32_t local_tid = idx.get_local_linear_id();
        uint32_t wg_id = idx.get_group(0);
        uint32_t wg_size = idx.get_local_range(0);
        uint32_t wg_count = idx.get_group_range(0);

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  _DataPerWorkItem = 256, BIN_COUNT = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change slm to reuse

        uint32_t slm_bin_hist_this_thread = local_tid * BIN_COUNT * sizeof(hist_t);
        uint32_t slm_lookup_subgroup = local_tid * sizeof(hist_t) * BIN_COUNT;

        simd<hist_t, BIN_COUNT> bin_offset;
        simd<hist_t, _DataPerWorkItem> ranks;
        simd<_KeyT, _DataPerWorkItem> keys;
        simd<bin_t, _DataPerWorkItem> bins;
        simd<device_addr_t, DATA_PER_STEP> lane_id(0, 1);

        device_addr_t io_offset = _DataPerWorkItem * (wg_id * wg_size + local_tid);
        constexpr _KeyT default_key = utils::__sort_identity<_KeyT, _IsAscending>();

        LoadKeys(io_offset, keys, default_key);

        bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<_IsAscending>(keys), stage * _RadixBits);

        ResetBinCounters(slm_bin_hist_this_thread);

        fence<fence_mask::local_barrier>();

        ranks = RankSLM(bins, slm_bin_hist_this_thread, local_tid);

        barrier();

        simd<hist_t, BIN_COUNT> subgroup_offset;
        simd<global_hist_t, BIN_COUNT> global_fix;

        global_hist_t* p_global_bin_start_buffer_allstages = reinterpret_cast<global_hist_t*>(p_global_buffer);
        global_hist_t* p_global_bin_start_buffer =
            p_global_bin_start_buffer_allstages + BIN_COUNT * STAGES + BIN_COUNT * wg_count * stage;

        global_hist_t* p_global_bin_this_group = p_global_bin_start_buffer + BIN_COUNT * wg_id;
        global_hist_t* p_global_bin_prev_group = p_global_bin_start_buffer + BIN_COUNT * (wg_id - 1);
        p_global_bin_prev_group = (0 == wg_id) ? (p_global_bin_start_buffer_allstages + BIN_COUNT * stage)
                                               : (p_global_bin_this_group - BIN_COUNT);

        UpdateGroupRank(local_tid, wg_id, subgroup_offset, global_fix, p_global_bin_prev_group,
                        p_global_bin_this_group);
        barrier();
        {
            bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<_IsAscending>(keys), stage * _RadixBits);

            slm_lookup_t<hist_t> subgroup_lookup(slm_lookup_subgroup);
            simd<hist_t, _DataPerWorkItem> wg_offset =
                ranks + subgroup_lookup.template lookup<_DataPerWorkItem>(subgroup_offset, bins);
            barrier();

            utils::VectorStore<_KeyT, 1, _DataPerWorkItem>(simd<uint32_t, _DataPerWorkItem>(wg_offset) * sizeof(_KeyT),
                                                           keys);
        }
        barrier();
        slm_lookup_t<global_hist_t> l(REORDER_SLM_SIZE);
        if (local_tid == 0)
        {
            l.template setup(global_fix);
        }
        barrier();
        {
            keys = utils::BlockLoad<_KeyT, _DataPerWorkItem>(local_tid * _DataPerWorkItem * sizeof(_KeyT));

            bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<_IsAscending>(keys), stage * _RadixBits);

            simd<hist_t, _DataPerWorkItem> group_offset =
                utils::create_simd<hist_t, _DataPerWorkItem>(local_tid * _DataPerWorkItem, 1);

            simd<device_addr_t, _DataPerWorkItem> global_offset =
                group_offset + l.template lookup<_DataPerWorkItem>(bins);

            utils::VectorStore<_KeyT, 1, _DataPerWorkItem>(output, global_offset * sizeof(_KeyT), keys,
                                                           global_offset < n);
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

template <typename _KeyT, ::std::uint32_t _RadixBits, ::std::uint32_t STAGES, ::std::uint32_t HW_TG_COUNT,
          ::std::uint32_t _WorkGroupSize, bool _IsAscending, typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <typename _KeyT, ::std::uint32_t _RadixBits, ::std::uint32_t STAGES, ::std::uint32_t HW_TG_COUNT,
          ::std::uint32_t _WorkGroupSize, bool _IsAscending, typename... _Name>
struct __radix_sort_onesweep_histogram_submitter<
    _KeyT, _RadixBits, STAGES, HW_TG_COUNT, _WorkGroupSize, _IsAscending,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _Range&& __rng, const _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(HW_TG_COUNT * _WorkGroupSize, _WorkGroupSize);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                __cgh.depends_on(__e);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        global_histogram<_KeyT, decltype(__data), _RadixBits, STAGES, HW_TG_COUNT, _WorkGroupSize,
                                         _IsAscending>(__nd_item, __n, __data, __global_offset_data);
                    });
            });
    }
};

template <::std::uint32_t STAGES, ::std::uint32_t BINCOUNT, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t STAGES, ::std::uint32_t BINCOUNT, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    STAGES, BINCOUNT, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(STAGES * BINCOUNT, BINCOUNT);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                __cgh.depends_on(__e);
                __cgh.parallel_for<_Name...>(__nd_range,
                                             [=](sycl::nd_item<1> __nd_item)
                                             {
                                                 uint32_t __offset = __nd_item.get_global_id(0);
                                                 auto __g = __nd_item.get_group();
                                                 uint32_t __count = __global_offset_data[__offset];
                                                 uint32_t __presum = sycl::exclusive_scan_over_group(
                                                     __g, __count, sycl::plus<::std::uint32_t>());
                                                 __global_offset_data[__offset] = __presum;
                                             });
            });
    }
};

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename... _Name>
struct __radix_sort_onesweep_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRange, typename _OutRange, typename _TmpData>
    sycl::event
    operator()(sycl::queue& __q, _InRange& __rng, _OutRange& __out_rng, const _TmpData& __tmp_data,
               ::std::uint32_t __sweep_tg_count, ::std::size_t __n, ::std::uint32_t __stage,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_tg_count * _WorkGroupSize, _WorkGroupSize);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng, __out_rng);
                auto __in_data = __rng.data();
                auto __out_data = __out_rng.data();
                __cgh.depends_on(__e);
                radix_sort_onesweep_slm_reorder_kernel<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                       _KeyT, decltype(__in_data), decltype(__out_data)>
                    K(__n, __stage, __in_data, __out_data, __tmp_data);
                __cgh.parallel_for<_Name...>(__nd_range, K);
            });
    }
};

template <typename _KeyT, typename _KernelName>
struct __radix_sort_copyback_submitter;

template <typename _KeyT, typename... _Name>
struct __radix_sort_copyback_submitter<_KeyT,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _TmpRange, typename _OutRange>
    sycl::event
    operator()(sycl::queue& __q, _TmpRange& __tmp_rng, _OutRange& __out_rng, ::std::uint32_t __n,
               const sycl::event& __e) const
    {
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __tmp_rng, __out_rng);
                // TODO: make sure that access is read_only for __tmp_data  and is write_only for __out_rng
                auto __tmp_data = __tmp_rng.data();
                auto __out_data = __out_rng.data();
                __cgh.depends_on(__e);
                __cgh.parallel_for<_Name...>(sycl::range<1>{__n},
                                             [=](sycl::item<1> __item)
                                             {
                                                 auto __global_id = __item.get_linear_id();
                                                 __out_data[__global_id] = __tmp_data[__global_id];
                                             });
            });
    }
};

template <typename _KernelName, bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _Range>
sycl::event
onesweep(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;

    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;

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
    constexpr uint32_t BINCOUNT = 1 << _RadixBits;

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr uint32_t _HistWorkGroupCount = 64;
    constexpr uint32_t _HistWorkGroupSize = 64;

    const uint32_t sweep_tg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, _WorkGroupSize * _DataPerWorkItem);
    constexpr uint32_t NBITS = sizeof(_KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, _RadixBits);

    // Memory for SYNC_BUFFER_SIZE is used by onesweep kernel implicitly
    // TODO: pass a pointer to the the memory allocated with SYNC_BUFFER_SIZE to onesweep kernel
    const uint32_t SYNC_BUFFER_SIZE = sweep_tg_count * BINCOUNT * STAGES * sizeof(global_hist_t); //bytes
    constexpr uint32_t GLOBAL_OFFSET_SIZE = BINCOUNT * STAGES * sizeof(global_hist_t);
    size_t temp_buffer_size = GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE;

    const size_t full_buffer_size_global_hist = temp_buffer_size * sizeof(uint8_t);
    const size_t full_buffer_size_output = __n * sizeof(_KeyT);
    const size_t full_buffer_size = full_buffer_size_global_hist + full_buffer_size_output;

    uint8_t* p_temp_memory = sycl::malloc_device<uint8_t>(full_buffer_size, __q);

    uint8_t* p_globl_hist_buffer = p_temp_memory;
    auto p_global_offset = reinterpret_cast<uint32_t*>(p_globl_hist_buffer);

    // Memory for storing values sorted for an iteration
    auto p_output = reinterpret_cast<_KeyT*>(p_temp_memory + full_buffer_size_global_hist);
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          decltype(p_output)>();
    auto __out_rng = __keep(p_output, p_output + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event event_chain = __q.memset(p_globl_hist_buffer, 0, temp_buffer_size);

    event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, _RadixBits, STAGES, _HistWorkGroupCount, _HistWorkGroupSize,
                                                  _IsAscending, _EsimdRadixSortHistogram>()(__q, __rng, p_global_offset,
                                                                                            __n, event_chain);

    event_chain = __radix_sort_onesweep_scan_submitter<STAGES, BINCOUNT, _EsimdRadixSortScan>()(__q, p_global_offset,
                                                                                                __n, event_chain);

    for (uint32_t stage = 0; stage < STAGES; stage++)
    {
        if ((stage % 2) == 0)
        {
            event_chain = __radix_sort_onesweep_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                          _KeyT, _EsimdRadixSortSweepEven>()(
                __q, __rng, __out_rng, p_globl_hist_buffer, sweep_tg_count, __n, stage, event_chain);
        }
        else
        {
            event_chain = __radix_sort_onesweep_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                          _KeyT, _EsimdRadixSortSweepOdd>()(
                __q, __out_rng, __rng, p_globl_hist_buffer, sweep_tg_count, __n, stage, event_chain);
        }
    }

    if constexpr (STAGES % 2 != 0)
    {
        event_chain =
            __radix_sort_copyback_submitter<_KeyT, _EsimdRadixSortCopyback>()(__q, __out_rng, __rng, __n, event_chain);
    }

    // TODO: required to remove this wait
    event_chain.wait();

    sycl::free(p_temp_memory, __q);

    return event_chain;
}

} // namespace oneapi::dpl::experimental::kt::esimd::impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
