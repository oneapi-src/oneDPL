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

namespace oneapi::dpl::experimental::kt::esimd::__impl
{
template <typename _KeyT, typename _InputT, ::std::uint32_t _RadixBits, ::std::uint32_t _STAGES, ::std::uint32_t _WORK_GROUPS,
          ::std::uint32_t _WORK_GROUP_SIZE, bool _IsAscending>
void
__global_histogram(sycl::nd_item<1> __idx, size_t __n, const _InputT& __input, ::std::uint32_t* __p_global_offset)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

    using _bin_t = ::std::uint16_t;
    using _hist_t = ::std::uint32_t;
    using _global_hist_t = ::std::uint32_t;

    slm_init(16384);

    constexpr ::std::uint32_t _BIN_COUNT = 1 << _RadixBits;
    constexpr ::std::uint32_t _DATA_PER_WORK_ITEM = 128;
    constexpr ::std::uint32_t _DEVICE_WIDE_STEP = _WORK_GROUPS * _WORK_GROUP_SIZE * _DATA_PER_WORK_ITEM;

    // Cap the number of histograms to reduce in SLM per __input range read pass
    // due to excessive GRF usage for thread-local histograms
    constexpr ::std::uint32_t _STAGES_PER_BLOCK = sizeof(_KeyT) < 4 ? sizeof(_KeyT) : 4;
    constexpr ::std::uint32_t _STAGE_BLOCKS = oneapi::dpl::__internal::__dpl_ceiling_div(_STAGES, _STAGES_PER_BLOCK);

    constexpr ::std::uint32_t _HIST_BUFFER_SIZE = _STAGES * _BIN_COUNT;
    constexpr ::std::uint32_t _HIST_DATA_PER_WORK_ITEM = _HIST_BUFFER_SIZE / _WORK_GROUP_SIZE;

    static_assert(_DATA_PER_WORK_ITEM % _DATA_PER_STEP == 0);
    static_assert(_BIN_COUNT * _STAGES_PER_BLOCK % _DATA_PER_STEP == 0);

    simd<_KeyT, _DATA_PER_WORK_ITEM> __keys;
    simd<_bin_t, _DATA_PER_WORK_ITEM> __bins;

    const ::std::uint32_t __local_id = __idx.get_local_linear_id();
    const ::std::uint32_t __global_id = __idx.get_global_linear_id();

    // 0. Early exit for threads without work
    if ((__global_id - __local_id) * _DATA_PER_WORK_ITEM > __n)
    {
        return;
    }

    // 1. Initialize group-local histograms in SLM
    __utils::_BlockStoreSlm<_global_hist_t, _HIST_DATA_PER_WORK_ITEM>(
        __local_id * _HIST_DATA_PER_WORK_ITEM * sizeof(_global_hist_t), 0);
    barrier();

#pragma unroll
    for (::std::uint32_t __stage_block = 0; __stage_block < _STAGE_BLOCKS; ++__stage_block)
    {
        simd<_global_hist_t, _BIN_COUNT * _STAGES_PER_BLOCK> __state_hist_grf(0);
        ::std::uint32_t __stage_block_start = __stage_block * _STAGES_PER_BLOCK;

        for (::std::uint32_t __wi_offset = __global_id * _DATA_PER_WORK_ITEM; __wi_offset < __n; __wi_offset += _DEVICE_WIDE_STEP)
        {
            // 1. Read __keys
            // TODO: avoid reading global memory twice when _STAGE_BLOCKS > 1 increasing _DATA_PER_WORK_ITEM
            if (__wi_offset + _DATA_PER_WORK_ITEM < __n)
            {
                __utils::__copy_from(__input, __wi_offset, __keys);
            }
            else
            {
                simd<::std::uint32_t, _DATA_PER_STEP> __lane_offsets(0, 1);
#pragma unroll
                for (::std::uint32_t __step_offset = 0; __step_offset < _DATA_PER_WORK_ITEM; __step_offset += _DATA_PER_STEP)
                {
                    simd<::std::uint32_t, _DATA_PER_STEP> __offsets = __lane_offsets + __step_offset + __wi_offset;
                    simd_mask<_DATA_PER_STEP> __is_in_range = __offsets < __n;
                    simd<_KeyT, _DATA_PER_STEP> data = __utils::__gather<_KeyT, _DATA_PER_STEP>(__input, __offsets, 0, __is_in_range);
                    simd<_KeyT, _DATA_PER_STEP> sort_identities = __utils::__sort_identity<_KeyT, _IsAscending>();
                    __keys.template select<_DATA_PER_STEP, 1>(__step_offset) = merge(data, sort_identities, __is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
#pragma unroll
            for (::std::uint32_t __stage_local = 0; __stage_local < _STAGES_PER_BLOCK; ++__stage_local)
            {
                constexpr _bin_t _MASK = _BIN_COUNT - 1;
                ::std::uint32_t __stage_global = __stage_block_start + __stage_local;
                __bins = __utils::__get_bucket<_MASK>(__utils::__order_preserving_cast<_IsAscending>(__keys),
                                                 __stage_global * _RadixBits);
#pragma unroll
                for (::std::uint32_t __i = 0; __i < _DATA_PER_WORK_ITEM; ++__i)
                {
                    ++__state_hist_grf[__stage_local * _BIN_COUNT + __bins[__i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
#pragma unroll
        for (::std::uint32_t __grf_offset = 0; __grf_offset < _BIN_COUNT * _STAGES_PER_BLOCK; __grf_offset += _DATA_PER_STEP)
        {
            ::std::uint32_t slm_offset = __stage_block_start * _BIN_COUNT + __grf_offset;
            simd<::std::uint32_t, _DATA_PER_STEP> __slm_byte_offsets(slm_offset * sizeof(_global_hist_t), sizeof(_global_hist_t));
            lsc_slm_atomic_update<atomic_op::add, _global_hist_t, _DATA_PER_STEP>(
                __slm_byte_offsets, __state_hist_grf.template select<_DATA_PER_STEP, 1>(__grf_offset), 1);
        }
        barrier();
    }

    // 4. Reduce group-local historgrams from SLM into global histograms in global memory
    simd<_global_hist_t, _HIST_DATA_PER_WORK_ITEM> __group_hist =
        __utils::_BlockLoadSlm<_global_hist_t, _HIST_DATA_PER_WORK_ITEM>(__local_id * _HIST_DATA_PER_WORK_ITEM *
                                                                    sizeof(_global_hist_t));
    simd<::std::uint32_t, _HIST_DATA_PER_WORK_ITEM> __byte_offsets(0, sizeof(_global_hist_t));
    lsc_atomic_update<atomic_op::add>(__p_global_offset + __local_id * _HIST_DATA_PER_WORK_ITEM, __byte_offsets, __group_hist,
                                      simd_mask<_HIST_DATA_PER_WORK_ITEM>(1));
}

template <::std::uint32_t _BITS>
inline __dpl_esimd_ns::simd<::std::uint32_t, 32>
__match_bins(__dpl_esimd_ns::simd<::std::uint32_t, 32> __bins, ::std::uint32_t __local_tid)
{
    //instruction count is 5*_BITS, so 40 for 8 bits.
    //performance is about 2 u per __bit for processing size 512 (will call this 16 times)
    // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
    // 8 bits is 12.8 us
    using namespace __dpl_esimd_ns;
    fence<fence_mask::sw_barrier>();
    simd<::std::uint32_t, 32> __matched_bins(0xffffffff);
#pragma unroll
    for (int __i = 0; __i < _BITS; __i++)
    {
        simd<::std::uint32_t, 32> __bit = (__bins >> __i) & 1;                                  // and
        simd<::std::uint32_t, 32> __x = __dpl_esimd_ns::merge<::std::uint32_t, 32>(0, -1, __bit != 0); // sel
        ::std::uint32_t __ones = pack_mask(__bit != 0);                                         // mov
        __matched_bins = __matched_bins & (__x ^ __ones);                                // bfn
    }
    fence<fence_mask::sw_barrier>();
    return __matched_bins;
}

template <typename _T>
struct _slm_lookup_t
{
    ::std::uint32_t __slm;
    inline _slm_lookup_t(::std::uint32_t __slm) : __slm(__slm) {}

    template <int _TABLE_SIZE>
    inline void
    setup(__dpl_esimd_ns::simd<_T, _TABLE_SIZE> __source) SYCL_ESIMD_FUNCTION
    {
        __utils::_BlockStoreSlm<_T, _TABLE_SIZE>(__slm, __source);
    }

    template <int _N, typename _IDX>
    inline auto
    lookup(_IDX __idx) SYCL_ESIMD_FUNCTION
    {
        return __utils::_VectorLoad<_T, 1, _N>(__slm + __dpl_esimd_ns::simd<::std::uint32_t, _N>(__idx) * sizeof(_T));
    }

    template <int _N, int _TABLE_SIZE, typename _IDX>
    inline auto
    lookup(__dpl_esimd_ns::simd<_T, _TABLE_SIZE> __source, _IDX __idx) SYCL_ESIMD_FUNCTION
    {
        setup(__source);
        return lookup<_N>(__idx);
    }
};

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename _InputT, typename _OutputT>
struct __radix_sort_onesweep_slm_reorder_kernel
{
    using _bin_t = ::std::uint16_t;
    using _hist_t = ::std::uint16_t;
    using _global_hist_t = ::std::uint32_t;
    using _device_addr_t = ::std::uint32_t;

    static constexpr ::std::uint32_t _BIN_COUNT = 1 << _RadixBits;
    static constexpr ::std::uint32_t _NBITS = sizeof(_KeyT) * 8;
    static constexpr ::std::uint32_t _STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(_NBITS, _RadixBits);
    static constexpr _bin_t _MASK = _BIN_COUNT - 1;
    static constexpr ::std::uint32_t _REORDER_SLM_SIZE = _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize;
    static constexpr ::std::uint32_t _BIN_HIST_SLM_SIZE = _BIN_COUNT * sizeof(_hist_t) * _WorkGroupSize;
    static constexpr ::std::uint32_t _SUBGROUP_LOOKUP_SIZE = _BIN_COUNT * sizeof(_hist_t) * _WorkGroupSize;
    static constexpr ::std::uint32_t _GLOBAL_LOOKUP_SIZE = _BIN_COUNT * sizeof(_global_hist_t);
    static constexpr ::std::uint32_t _INCOMING_OFFSET_SLM_SIZE = (_BIN_COUNT+1)*sizeof(_hist_t);
    static constexpr ::std::uint32_t _OFFSETS_SLM_SIZE = _BIN_HIST_SLM_SIZE + _GLOBAL_LOOKUP_SIZE + _INCOMING_OFFSET_SLM_SIZE;

    // slm allocation:
    // first stage, do subgroup ranks, need _WorkGroupSize*_BIN_COUNT*sizeof(_hist_t)
    // then do rank roll up in workgroup, need _WorkGroupSize*_BIN_COUNT*sizeof(_hist_t) + _BIN_COUNT*sizeof(_hist_t) + _BIN_COUNT*sizeof(_global_hist_t)
    // after all these is done, update ranks to workgroup ranks, need _SUBGROUP_LOOKUP_SIZE
    // then shuffle keys to workgroup order in SLM, need _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize
    // then read reordered slm and look up global fix, need _GLOBAL_LOOKUP_SIZE on top

    const ::std::uint32_t __n;
    const ::std::uint32_t __stage;
    _InputT __input;
    _OutputT __output;
    ::std::uint8_t* __p_global_buffer;

    __radix_sort_onesweep_slm_reorder_kernel(::std::uint32_t __n, ::std::uint32_t __stage, _InputT __input, _OutputT __output,
                                           ::std::uint8_t* __p_global_buffer)
        : __n(__n), __stage(__stage), __input(__input), __output(__output), __p_global_buffer(__p_global_buffer)
    {
    }

    inline void
    _LoadKeys(::std::uint32_t __io_offset, __dpl_esimd_ns::simd<_KeyT, _DataPerWorkItem>& __keys, _KeyT __default_key) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        static_assert(_DataPerWorkItem % _DATA_PER_STEP == 0);

        bool __is_full_block = (__io_offset + _DataPerWorkItem) < __n;
        if (__is_full_block)
        {
            simd<::std::uint32_t, _DATA_PER_STEP> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __keys.template select<_DATA_PER_STEP, 1>(__s) = __utils::__gather<_KeyT, _DATA_PER_STEP>(__input, __offset, 0);
            }
        }
        else
        {
            simd<::std::uint32_t, _DATA_PER_STEP> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                simd_mask<_DATA_PER_STEP> __m = __offset < __n;
                __keys.template select<_DATA_PER_STEP, 1>(__s) = merge(__utils::__gather<_KeyT, _DATA_PER_STEP>(__input, __offset, 0, __m),
                                                                  simd<_KeyT, _DATA_PER_STEP>(__default_key), __m);
            }
        }
    }

    inline void
    _ResetBinCounters(::std::uint32_t __slm_bin_hist_this_thread) const
    {
        __utils::_BlockStoreSlm<_hist_t, _BIN_COUNT>(__slm_bin_hist_this_thread, 0);
    }

    inline auto
    RankSLM(__dpl_esimd_ns::simd<_bin_t, _DataPerWorkItem> __bins, ::std::uint32_t __slm_counter_offset, ::std::uint32_t __local_tid) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        constexpr int _BinsPerStep = 32;
        static_assert(_DataPerWorkItem % _BinsPerStep == 0);

        constexpr ::std::uint32_t _BIN_COUNT = 1 << _RadixBits;
        simd<::std::uint32_t, _DataPerWorkItem> __ranks;
        __utils::_BlockStoreSlm<_hist_t, _BIN_COUNT>(__slm_counter_offset, 0);
        simd<::std::uint32_t, _BinsPerStep> __remove_right_lanes, __lane_id(0, 1);
        __remove_right_lanes = 0x7fffffff >> (_BinsPerStep - 1 - __lane_id);
#pragma unroll
        for (::std::uint32_t __s = 0; __s < _DataPerWorkItem; __s += _BinsPerStep)
        {
            simd<::std::uint32_t, _BinsPerStep> __this_bins = __bins.template select<_BinsPerStep, 1>(__s);
            simd<::std::uint32_t, _BinsPerStep> __matched_bins = __match_bins<_RadixBits>(__this_bins, __local_tid); // 40 insts
            simd<::std::uint32_t, _BinsPerStep> __pre_rank, __this_round_rank;
            __pre_rank = __utils::_VectorLoad<_hist_t, 1, _BinsPerStep>(__slm_counter_offset +
                                                                 __this_bins * sizeof(_hist_t)); // 2 mad+load.__slm
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            __this_round_rank = cbit(__matched_left_lanes);
            auto __this_round_count = cbit(__matched_bins);
            auto __rank_after = __pre_rank + __this_round_rank;
            auto __is_leader = __this_round_rank == __this_round_count - 1;
            __utils::_VectorStore<_hist_t, 1, _BinsPerStep>(__slm_counter_offset + __this_bins * sizeof(_hist_t), __rank_after + 1,
                                                       __is_leader);
            __ranks.template select<_BinsPerStep, 1>(__s) = __rank_after;
        }
        return __ranks;
    }

    inline void
    _UpdateGroupRank(::std::uint32_t __local_tid, ::std::uint32_t __wg_id, __dpl_esimd_ns::simd<_hist_t, _BIN_COUNT>& __subgroup_offset,
                    __dpl_esimd_ns::simd<_global_hist_t, _BIN_COUNT>& __global_fix, _global_hist_t* __p_global_bin_prev_group,
                    _global_hist_t* __p_global_bin_this_group) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        /*
        first do column scan by group, each thread do 32c,
        then last row do exclusive scan as group incoming __offset
        then every thread add local sum with sum of previous group and incoming __offset
        */
        constexpr ::std::uint32_t _HIST_STRIDE = sizeof(_hist_t) * _BIN_COUNT;
        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * _HIST_STRIDE;
        const ::std::uint32_t __slm_bin_hist_group_incoming = _WorkGroupSize * _HIST_STRIDE;
        const ::std::uint32_t __slm_bin_hist_global_incoming = __slm_bin_hist_group_incoming + _HIST_STRIDE;
        constexpr ::std::uint32_t _GLOBAL_ACCUMULATED = 0x40000000;
        constexpr ::std::uint32_t _HIST_UPDATED = 0x80000000;
        constexpr ::std::uint32_t _GLOBAL_OFFSET_MASK = 0x3fffffff;
        {
            barrier();
            constexpr ::std::uint32_t _BIN_SUMMARY_GROUP_SIZE = 8;
            constexpr ::std::uint32_t _BIN_WIDTH = _BIN_COUNT / _BIN_SUMMARY_GROUP_SIZE;

            static_assert(_BIN_COUNT % _BIN_WIDTH == 0);

            simd<_hist_t, _BIN_WIDTH> __thread_grf_hist_summary(0);
            if (__local_tid < _BIN_SUMMARY_GROUP_SIZE)
            {
                ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * _BIN_WIDTH * sizeof(_hist_t);
                for (::std::uint32_t __s = 0; __s < _WorkGroupSize; __s++, __slm_bin_hist_summary_offset += _HIST_STRIDE)
                {
                    __thread_grf_hist_summary += __utils::_BlockLoadSlm<_hist_t, _BIN_WIDTH>(__slm_bin_hist_summary_offset);
                    __utils::_BlockStoreSlm(__slm_bin_hist_summary_offset, __thread_grf_hist_summary);
                }

                __utils::_BlockStoreSlm(__slm_bin_hist_group_incoming + __local_tid * _BIN_WIDTH * sizeof(_hist_t),
                                     __utils::__scan<_hist_t, _hist_t>(__thread_grf_hist_summary));
                if (__wg_id != 0)
                    __utils::_BlockStore<::std::uint32_t, _BIN_WIDTH>(__p_global_bin_this_group + __local_tid * _BIN_WIDTH,
                                                           __thread_grf_hist_summary | _HIST_UPDATED);
            }
            barrier();
            if (__local_tid == _BIN_SUMMARY_GROUP_SIZE + 1)
            {
                // this thread to group scan
                simd<_hist_t, _BIN_COUNT> __grf_hist_summary;
                simd<_hist_t, _BIN_COUNT + 1> __grf_hist_summary_scan;
                __grf_hist_summary = __utils::_BlockLoadSlm<_hist_t, _BIN_COUNT>(__slm_bin_hist_group_incoming);
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<_BIN_WIDTH, 1>(1) =
                    __grf_hist_summary.template select<_BIN_WIDTH, 1>(0);
#pragma unroll
                for (::std::uint32_t __i = _BIN_WIDTH; __i < _BIN_COUNT; __i += _BIN_WIDTH)
                {
                    __grf_hist_summary_scan.template select<_BIN_WIDTH, 1>(__i + 1) =
                        __grf_hist_summary.template select<_BIN_WIDTH, 1>(__i) + __grf_hist_summary_scan[__i];
                }
                __utils::_BlockStoreSlm<_hist_t, _BIN_COUNT>(__slm_bin_hist_group_incoming,
                                                        __grf_hist_summary_scan.template select<_BIN_COUNT, 1>());
            }
            else if (__local_tid < _BIN_SUMMARY_GROUP_SIZE)
            {
                // these threads to global sync and update
                simd<_global_hist_t, _BIN_WIDTH> __prev_group_hist_sum(0), __prev_group_hist;
                simd_mask<_BIN_WIDTH> __is_not_accumulated(1);
                do
                {
                    do
                    {
                        __prev_group_hist =
                            lsc_block_load<_global_hist_t, _BIN_WIDTH, lsc_data_size::default_size, cache_hint::uncached,
                                           cache_hint::cached>(__p_global_bin_prev_group + __local_tid * _BIN_WIDTH);
                        fence<fence_mask::sw_barrier>();
                    } while (((__prev_group_hist & _HIST_UPDATED) == 0).any() && __wg_id != 0);
                    __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                    __is_not_accumulated = (__prev_group_hist_sum & _GLOBAL_ACCUMULATED) == 0;
                    __p_global_bin_prev_group -= _BIN_COUNT;
                } while (__is_not_accumulated.any() && __wg_id != 0);
                __prev_group_hist_sum &= _GLOBAL_OFFSET_MASK;
                simd<_global_hist_t, _BIN_WIDTH> after_group_hist_sum = __prev_group_hist_sum + __thread_grf_hist_summary;
                __utils::_BlockStore<::std::uint32_t, _BIN_WIDTH>(__p_global_bin_this_group + __local_tid * _BIN_WIDTH,
                                                       after_group_hist_sum | _HIST_UPDATED | _GLOBAL_ACCUMULATED);

                __utils::_BlockStoreSlm<::std::uint32_t, _BIN_WIDTH>(
                    __slm_bin_hist_global_incoming + __local_tid * _BIN_WIDTH * sizeof(_global_hist_t), __prev_group_hist_sum);
            }
            barrier();
        }
        auto __group_incoming = __utils::_BlockLoadSlm<_hist_t, _BIN_COUNT>(__slm_bin_hist_group_incoming);
        __global_fix = __utils::_BlockLoadSlm<_global_hist_t, _BIN_COUNT>(__slm_bin_hist_global_incoming) - __group_incoming;
        if (__local_tid > 0)
        {
            __subgroup_offset = __group_incoming + __utils::_BlockLoadSlm<_hist_t, _BIN_COUNT>((__local_tid - 1) * _HIST_STRIDE);
        }
        else
            __subgroup_offset = __group_incoming;
    }

    void
    operator()(sycl::nd_item<1> __idx) const SYCL_ESIMD_KERNEL
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        slm_init(128 * 1024);

        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_id = __idx.get_group(0);
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __wg_count = __idx.get_group_range(0);

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  _DataPerWorkItem = 256, _BIN_COUNT = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change __slm to reuse

        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * _BIN_COUNT * sizeof(_hist_t);
        const ::std::uint32_t __slm_lookup_subgroup = __local_tid * sizeof(_hist_t) * _BIN_COUNT;

        simd<_hist_t, _BIN_COUNT> __bin_offset;
        simd<_hist_t, _DataPerWorkItem> __ranks;
        simd<_KeyT, _DataPerWorkItem> __keys;
        simd<_bin_t, _DataPerWorkItem> __bins;
        simd<_device_addr_t, _DATA_PER_STEP> __lane_id(0, 1);

        const _device_addr_t __io_offset = _DataPerWorkItem * (__wg_id * __wg_size + __local_tid);
        constexpr _KeyT __default_key = __utils::__sort_identity<_KeyT, _IsAscending>();

        _LoadKeys(__io_offset, __keys, __default_key);

        __bins = __utils::__get_bucket<_MASK>(__utils::__order_preserving_cast<_IsAscending>(__keys), __stage * _RadixBits);

        _ResetBinCounters(__slm_bin_hist_this_thread);

        fence<fence_mask::local_barrier>();

        __ranks = RankSLM(__bins, __slm_bin_hist_this_thread, __local_tid);

        barrier();

        simd<_hist_t, _BIN_COUNT> __subgroup_offset;
        simd<_global_hist_t, _BIN_COUNT> __global_fix;

        _global_hist_t* __p_global_bin_start_buffer_allstages = reinterpret_cast<_global_hist_t*>(__p_global_buffer);
        _global_hist_t* __p_global_bin_start_buffer =
            __p_global_bin_start_buffer_allstages + _BIN_COUNT * _STAGES + _BIN_COUNT * __wg_count * __stage;

        _global_hist_t* __p_global_bin_this_group = __p_global_bin_start_buffer + _BIN_COUNT * __wg_id;
        _global_hist_t* __p_global_bin_prev_group = __p_global_bin_start_buffer + _BIN_COUNT * (__wg_id - 1);
        __p_global_bin_prev_group = (0 == __wg_id) ? (__p_global_bin_start_buffer_allstages + _BIN_COUNT * __stage)
                                               : (__p_global_bin_this_group - _BIN_COUNT);

        _UpdateGroupRank(__local_tid, __wg_id, __subgroup_offset, __global_fix, __p_global_bin_prev_group,
                        __p_global_bin_this_group);
        barrier();
        {
            __bins = __utils::__get_bucket<_MASK>(__utils::__order_preserving_cast<_IsAscending>(__keys), __stage * _RadixBits);

            _slm_lookup_t<_hist_t> __subgroup_lookup(__slm_lookup_subgroup);
            simd<_hist_t, _DataPerWorkItem> __wg_offset =
                __ranks + __subgroup_lookup.template lookup<_DataPerWorkItem>(__subgroup_offset, __bins);
            barrier();

            __utils::_VectorStore<_KeyT, 1, _DataPerWorkItem>(simd<::std::uint32_t, _DataPerWorkItem>(__wg_offset) * sizeof(_KeyT),
                                                           __keys);
        }
        barrier();
        _slm_lookup_t<_global_hist_t> __global_fix_lookup(_REORDER_SLM_SIZE);
        if (__local_tid == 0)
        {
            __global_fix_lookup.template setup(__global_fix);
        }
        barrier();
        {
            __keys = __utils::_BlockLoadSlm<_KeyT, _DataPerWorkItem>(__local_tid * _DataPerWorkItem * sizeof(_KeyT));

            __bins = __utils::__get_bucket<_MASK>(__utils::__order_preserving_cast<_IsAscending>(__keys), __stage * _RadixBits);

            simd<_hist_t, _DataPerWorkItem> __group_offset =
                __utils::__create_simd<_hist_t, _DataPerWorkItem>(__local_tid * _DataPerWorkItem, 1);

            simd<_device_addr_t, _DataPerWorkItem> __global_offset =
                __group_offset + __global_fix_lookup.template lookup<_DataPerWorkItem>(__bins);

            __utils::_VectorStore<_KeyT, 1, _DataPerWorkItem>(__output, __global_offset * sizeof(_KeyT), __keys,
                                                           __global_offset < __n);
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

template <typename _KeyT, ::std::uint32_t _RadixBits, ::std::uint32_t _STAGES, ::std::uint32_t _HW_TG_COUNT,
          ::std::uint32_t _WorkGroupSize, bool _IsAscending, typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <typename _KeyT, ::std::uint32_t _RadixBits, ::std::uint32_t _STAGES, ::std::uint32_t _HW_TG_COUNT,
          ::std::uint32_t _WorkGroupSize, bool _IsAscending, typename... _Name>
struct __radix_sort_onesweep_histogram_submitter<
    _KeyT, _RadixBits, _STAGES, _HW_TG_COUNT, _WorkGroupSize, _IsAscending,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _Range&& __rng, const _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(_HW_TG_COUNT * _WorkGroupSize, _WorkGroupSize);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                __cgh.depends_on(__e);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        __global_histogram<_KeyT, decltype(__data), _RadixBits, _STAGES, _HW_TG_COUNT, _WorkGroupSize,
                                         _IsAscending>(__nd_item, __n, __data, __global_offset_data);
                    });
            });
    }
};

template <::std::uint32_t _STAGES, ::std::uint32_t _BINCOUNT, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t _STAGES, ::std::uint32_t _BINCOUNT, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    _STAGES, _BINCOUNT, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(_STAGES * _BINCOUNT, _BINCOUNT);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                __cgh.depends_on(__e);
                __cgh.parallel_for<_Name...>(__nd_range,
                                             [=](sycl::nd_item<1> __nd_item)
                                             {
                                                 ::std::uint32_t __offset = __nd_item.get_global_id(0);
                                                 const auto __g = __nd_item.get_group();
                                                 ::std::uint32_t __count = __global_offset_data[__offset];
                                                 ::std::uint32_t __presum = sycl::exclusive_scan_over_group(
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
                __radix_sort_onesweep_slm_reorder_kernel<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                       _KeyT, decltype(__in_data), decltype(__out_data)>
                    __sweep_kernel(__n, __stage, __in_data, __out_data, __tmp_data);
                __cgh.parallel_for<_Name...>(__nd_range, __sweep_kernel);
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
__onesweep(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
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

    using _global_hist_t = ::std::uint32_t;
    constexpr ::std::uint32_t _BINCOUNT = 1 << _RadixBits;

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr ::std::uint32_t _HistWorkGroupCount = 64;
    constexpr ::std::uint32_t _HistWorkGroupSize = 64;

    const ::std::uint32_t __sweep_tg_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, _WorkGroupSize * _DataPerWorkItem);
    constexpr ::std::uint32_t _NBITS = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t _STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(_NBITS, _RadixBits);

    // Memory for _SYNC_BUFFER_SIZE is used by onesweep kernel implicitly
    // TODO: pass a pointer to the the memory allocated with _SYNC_BUFFER_SIZE to onesweep kernel
    const ::std::uint32_t _SYNC_BUFFER_SIZE = __sweep_tg_count * _BINCOUNT * _STAGES * sizeof(_global_hist_t); //bytes
    constexpr ::std::uint32_t _GLOBAL_OFFSET_SIZE = _BINCOUNT * _STAGES * sizeof(_global_hist_t);
    size_t __temp_buffer_size = _GLOBAL_OFFSET_SIZE + _SYNC_BUFFER_SIZE;

    const size_t __full_buffer_size_global_hist = __temp_buffer_size * sizeof(::std::uint8_t);
    const size_t __full_buffer_size_output = __n * sizeof(_KeyT);
    const size_t __full_buffer_size = __full_buffer_size_global_hist + __full_buffer_size_output;

    ::std::uint8_t* __p_temp_memory = sycl::malloc_device<::std::uint8_t>(__full_buffer_size, __q);

    ::std::uint8_t* __p_globl_hist_buffer = __p_temp_memory;
    auto __p_global_offset = reinterpret_cast<::std::uint32_t*>(__p_globl_hist_buffer);

    // Memory for storing values sorted for an iteration
    auto _p_output = reinterpret_cast<_KeyT*>(__p_temp_memory + __full_buffer_size_global_hist);
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          decltype(_p_output)>();
    auto __out_rng = __keep(_p_output, _p_output + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event __event_chain = __q.memset(__p_globl_hist_buffer, 0, __temp_buffer_size);

    __event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, _RadixBits, _STAGES, _HistWorkGroupCount, _HistWorkGroupSize,
                                                  _IsAscending, _EsimdRadixSortHistogram>()(__q, __rng, __p_global_offset,
                                                                                            __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<_STAGES, _BINCOUNT, _EsimdRadixSortScan>()(__q, __p_global_offset,
                                                                                                __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < _STAGES; __stage++)
    {
        if ((__stage % 2) == 0)
        {
            __event_chain = __radix_sort_onesweep_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                          _KeyT, _EsimdRadixSortSweepEven>()(
                __q, __rng, __out_rng, __p_globl_hist_buffer, __sweep_tg_count, __n, __stage, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize,
                                                          _KeyT, _EsimdRadixSortSweepOdd>()(
                __q, __out_rng, __rng, __p_globl_hist_buffer, __sweep_tg_count, __n, __stage, __event_chain);
        }
    }

    if constexpr (_STAGES % 2 != 0)
    {
        __event_chain =
            __radix_sort_copyback_submitter<_KeyT, _EsimdRadixSortCopyback>()(__q, __out_rng, __rng, __n, __event_chain);
    }

    __event_chain = __q.submit(
        [__event_chain, __p_temp_memory, __q](sycl::handler& __cgh)
        {
            __cgh.depends_on(__event_chain);
            __cgh.host_task([&]() { sycl::free(__p_temp_memory, __q); });
        });

    return __event_chain;
}

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
