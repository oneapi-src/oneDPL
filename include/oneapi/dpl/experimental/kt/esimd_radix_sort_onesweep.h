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
template <typename _KeyT, typename _InputT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count, ::std::uint32_t __hist_work_group_count,
          ::std::uint32_t __hist_work_group_size, bool __is_ascending>
void
__global_histogram(sycl::nd_item<1> __idx, size_t __n, const _InputT& __input, ::std::uint32_t* __p_global_offset)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint32_t;
    using _GlobalHistT = ::std::uint32_t;

    slm_init(16384);

    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    constexpr ::std::uint32_t __hist_data_per_work_item = 128;
    constexpr ::std::uint32_t __device_wide_step = __hist_work_group_count * __hist_work_group_size * __hist_data_per_work_item;

    // Cap the number of histograms to reduce in SLM per __input range read pass
    // due to excessive GRF usage for thread-local histograms
    constexpr ::std::uint32_t __stages_per_block = sizeof(_KeyT) < 4 ? sizeof(_KeyT) : 4;
    constexpr ::std::uint32_t __stage_block_count = oneapi::dpl::__internal::__dpl_ceiling_div(__stage_count, __stages_per_block);

    constexpr ::std::uint32_t __hist_buffer_size = __stage_count * __bin_count;
    constexpr ::std::uint32_t __group_hist_size = __hist_buffer_size / __hist_work_group_size;

    static_assert(__hist_data_per_work_item % __data_per_step == 0);
    static_assert(__bin_count * __stages_per_block % __data_per_step == 0);

    simd<_KeyT, __hist_data_per_work_item> __keys;
    simd<_BinT, __hist_data_per_work_item> __bins;

    const ::std::uint32_t __local_id = __idx.get_local_linear_id();
    const ::std::uint32_t __global_id = __idx.get_global_linear_id();

    // 0. Early exit for threads without work
    if ((__global_id - __local_id) * __hist_data_per_work_item > __n)
    {
        return;
    }

    // 1. Initialize group-local histograms in SLM
    __utils::__block_store_slm<_GlobalHistT, __group_hist_size>(
        __local_id * __group_hist_size * sizeof(_GlobalHistT), 0);
    barrier();

#pragma unroll
    for (::std::uint32_t __stage_block = 0; __stage_block < __stage_block_count; ++__stage_block)
    {
        simd<_GlobalHistT, __bin_count * __stages_per_block> __state_hist_grf(0);
        ::std::uint32_t __stage_block_start = __stage_block * __stages_per_block;

        for (::std::uint32_t __wi_offset = __global_id * __hist_data_per_work_item; __wi_offset < __n; __wi_offset += __device_wide_step)
        {
            // 1. Read __keys
            // TODO: avoid reading global memory twice when __stage_block_count > 1 increasing __hist_data_per_work_item
            if (__wi_offset + __hist_data_per_work_item < __n)
            {
                __utils::__copy_from(__input, __wi_offset, __keys);
            }
            else
            {
                simd<::std::uint32_t, __data_per_step> __lane_offsets(0, 1);
#pragma unroll
                for (::std::uint32_t __step_offset = 0; __step_offset < __hist_data_per_work_item; __step_offset += __data_per_step)
                {
                    simd<::std::uint32_t, __data_per_step> __offsets = __lane_offsets + __step_offset + __wi_offset;
                    simd_mask<__data_per_step> __is_in_range = __offsets < __n;
                    simd<_KeyT, __data_per_step> data = __utils::__gather<_KeyT, __data_per_step>(__input, __offsets, 0, __is_in_range);
                    simd<_KeyT, __data_per_step> sort_identities = __utils::__sort_identity<_KeyT, __is_ascending>();
                    __keys.template select<__data_per_step, 1>(__step_offset) = merge(data, sort_identities, __is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
#pragma unroll
            for (::std::uint32_t __stage_local = 0; __stage_local < __stages_per_block; ++__stage_local)
            {
                constexpr _BinT __mask = __bin_count - 1;
                ::std::uint32_t __stage_global = __stage_block_start + __stage_local;
                __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys),
                                                 __stage_global * __radix_bits);
#pragma unroll
                for (::std::uint32_t __i = 0; __i < __hist_data_per_work_item; ++__i)
                {
                    ++__state_hist_grf[__stage_local * __bin_count + __bins[__i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
#pragma unroll
        for (::std::uint32_t __grf_offset = 0; __grf_offset < __bin_count * __stages_per_block; __grf_offset += __data_per_step)
        {
            ::std::uint32_t slm_offset = __stage_block_start * __bin_count + __grf_offset;
            simd<::std::uint32_t, __data_per_step> __slm_byte_offsets(slm_offset * sizeof(_GlobalHistT), sizeof(_GlobalHistT));
            lsc_slm_atomic_update<atomic_op::add, _GlobalHistT, __data_per_step>(
                __slm_byte_offsets, __state_hist_grf.template select<__data_per_step, 1>(__grf_offset), 1);
        }
        barrier();
    }

    // 4. Reduce group-local historgrams from SLM into global histograms in global memory
    simd<_GlobalHistT, __group_hist_size> __group_hist =
        __utils::__block_load_slm<_GlobalHistT, __group_hist_size>(__local_id * __group_hist_size *
                                                                    sizeof(_GlobalHistT));
    simd<::std::uint32_t, __group_hist_size> __byte_offsets(0, sizeof(_GlobalHistT));
    lsc_atomic_update<atomic_op::add>(__p_global_offset + __local_id * __group_hist_size, __byte_offsets, __group_hist,
                                      simd_mask<__group_hist_size>(1));
}

template <::std::uint32_t __bit_count>
inline __dpl_esimd_ns::simd<::std::uint32_t, 32>
__match_bins(__dpl_esimd_ns::simd<::std::uint32_t, 32> __bins, ::std::uint32_t __local_tid)
{
    //instruction count is 5*__bit_count, so 40 for 8 bits.
    //performance is about 2 u per __bit for processing size 512 (will call this 16 times)
    // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
    // 8 bits is 12.8 us
    using namespace __dpl_esimd_ns;
    fence<fence_mask::sw_barrier>();
    simd<::std::uint32_t, 32> __matched_bins(0xffffffff);
#pragma unroll
    for (int __i = 0; __i < __bit_count; __i++)
    {
        simd<::std::uint32_t, 32> __bit = (__bins >> __i) & 1;                                         // and
        simd<::std::uint32_t, 32> __x = __dpl_esimd_ns::merge<::std::uint32_t, 32>(0, -1, __bit != 0); // sel
        ::std::uint32_t __ones = pack_mask(__bit != 0);                                                // mov
        __matched_bins = __matched_bins & (__x ^ __ones);                                              // bfn
    }
    fence<fence_mask::sw_barrier>();
    return __matched_bins;
}

template <typename _T>
struct _slm_lookup_t
{
    ::std::uint32_t __slm;
    inline _slm_lookup_t(::std::uint32_t __slm) : __slm(__slm) {}

    template <int __table_size>
    inline void
    __setup(__dpl_esimd_ns::simd<_T, __table_size> __source) SYCL_ESIMD_FUNCTION
    {
        __utils::__block_store_slm<_T, __table_size>(__slm, __source);
    }

    template <int _N, typename _Idx>
    inline auto
    __lookup(_Idx __idx) SYCL_ESIMD_FUNCTION
    {
        return __utils::__vector_load<_T, 1, _N>(__slm + __dpl_esimd_ns::simd<::std::uint32_t, _N>(__idx) * sizeof(_T));
    }

    template <int _N, int __table_size, typename _Idx>
    inline auto
    __lookup(__dpl_esimd_ns::simd<_T, __table_size> __source, _Idx __idx) SYCL_ESIMD_FUNCTION
    {
        __setup(__source);
        return __lookup<_N>(__idx);
    }
};

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _InputT, typename _OutputT>
struct __radix_sort_onesweep_slm_reorder_kernel
{
    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint16_t;
    using _GlobalHistT = ::std::uint32_t;
    using _DeviceAddrT = ::std::uint32_t;

    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
    static constexpr _BinT __mask = __bin_count - 1;
    static constexpr ::std::uint32_t __reorder_slm_size = __data_per_work_item * sizeof(_KeyT) * __work_group_size;
    static constexpr ::std::uint32_t __bin_hist_slm_size = __bin_count * sizeof(_HistT) * __work_group_size;
    static constexpr ::std::uint32_t __sub_group_lookup_size = __bin_count * sizeof(_HistT) * __work_group_size;
    static constexpr ::std::uint32_t __global_lookup_size = __bin_count * sizeof(_GlobalHistT);
    static constexpr ::std::uint32_t __incoming_offset_slm_size = (__bin_count+1)*sizeof(_HistT);

    // slm allocation:
    // first stage, do subgroup ranks, need __work_group_size*__bin_count*sizeof(_HistT)
    // then do rank roll up in workgroup, need __work_group_size*__bin_count*sizeof(_HistT) + __bin_count*sizeof(_HistT) + __bin_count*sizeof(_GlobalHistT)
    // after all these is done, update ranks to workgroup ranks, need __sub_group_lookup_size
    // then shuffle keys to workgroup order in SLM, need __data_per_work_item * sizeof(_KeyT) * __work_group_size
    // then read reordered slm and look up global fix, need __global_lookup_size on top

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
    __load_keys(::std::uint32_t __io_offset, __dpl_esimd_ns::simd<_KeyT, __data_per_work_item>& __keys, _KeyT __default_key) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        static_assert(__data_per_work_item % __data_per_step == 0);

        bool __is_full_block = (__io_offset + __data_per_work_item) < __n;
        if (__is_full_block)
        {
            simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __keys.template select<__data_per_step, 1>(__s) = __utils::__gather<_KeyT, __data_per_step>(__input, __offset, 0);
            }
        }
        else
        {
            simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                simd_mask<__data_per_step> __m = __offset < __n;
                __keys.template select<__data_per_step, 1>(__s) = merge(__utils::__gather<_KeyT, __data_per_step>(__input, __offset, 0, __m),
                                                                  simd<_KeyT, __data_per_step>(__default_key), __m);
            }
        }
    }

    inline void
    __reset_bin_counters(::std::uint32_t __slm_bin_hist_this_thread) const
    {
        __utils::__block_store_slm<_HistT, __bin_count>(__slm_bin_hist_this_thread, 0);
    }

    inline auto
    __rank_slm(__dpl_esimd_ns::simd<_BinT, __data_per_work_item> __bins, ::std::uint32_t __slm_counter_offset, ::std::uint32_t __local_tid) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        constexpr int __bins_per_step = 32;
        static_assert(__data_per_work_item % __bins_per_step == 0);

        constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
        simd<::std::uint32_t, __data_per_work_item> __ranks;
        __utils::__block_store_slm<_HistT, __bin_count>(__slm_counter_offset, 0);
        simd<::std::uint32_t, __bins_per_step> __remove_right_lanes, __lane_id(0, 1);
        __remove_right_lanes = 0x7fffffff >> (__bins_per_step - 1 - __lane_id);
#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __bins_per_step)
        {
            simd<::std::uint32_t, __bins_per_step> __this_bins = __bins.template select<__bins_per_step, 1>(__s);
            simd<::std::uint32_t, __bins_per_step> __matched_bins = __match_bins<__radix_bits>(__this_bins, __local_tid); // 40 insts
            simd<::std::uint32_t, __bins_per_step> __pre_rank, __this_round_rank;
            __pre_rank = __utils::__vector_load<_HistT, 1, __bins_per_step>(__slm_counter_offset +
                                                                 __this_bins * sizeof(_HistT)); // 2 mad+load.__slm
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            __this_round_rank = cbit(__matched_left_lanes);
            auto __this_round_count = cbit(__matched_bins);
            auto __rank_after = __pre_rank + __this_round_rank;
            auto __is_leader = __this_round_rank == __this_round_count - 1;
            __utils::__vector_store<_HistT, 1, __bins_per_step>(__slm_counter_offset + __this_bins * sizeof(_HistT), __rank_after + 1,
                                                       __is_leader);
            __ranks.template select<__bins_per_step, 1>(__s) = __rank_after;
        }
        return __ranks;
    }

    inline void
    __update_group_rank(::std::uint32_t __local_tid, ::std::uint32_t __wg_id, __dpl_esimd_ns::simd<_HistT, __bin_count>& __subgroup_offset,
                    __dpl_esimd_ns::simd<_GlobalHistT, __bin_count>& __global_fix, _GlobalHistT* __p_global_bin_prev_group,
                    _GlobalHistT* __p_global_bin_this_group) const
    {
        using namespace __dpl_esimd_ns;
        using namespace __dpl_esimd_ens;

        /*
        first do column scan by group, each thread do 32c,
        then last row do exclusive scan as group incoming __offset
        then every thread add local sum with sum of previous group and incoming __offset
        */
        constexpr ::std::uint32_t __hist_stride = sizeof(_HistT) * __bin_count;
        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __hist_stride;
        const ::std::uint32_t __slm_bin_hist_group_incoming = __work_group_size * __hist_stride;
        const ::std::uint32_t __slm_bin_hist_global_incoming = __slm_bin_hist_group_incoming + __hist_stride;
        constexpr ::std::uint32_t __global_accumulated = 0x40000000;
        constexpr ::std::uint32_t __hist_updated = 0x80000000;
        constexpr ::std::uint32_t __global_offset_mask = 0x3fffffff;
        {
            barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;

            static_assert(__bin_count % __bin_width == 0);

            simd<_HistT, __bin_width> __thread_grf_hist_summary(0);
            if (__local_tid < __bin_summary_group_size)
            {
                ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * __bin_width * sizeof(_HistT);
                for (::std::uint32_t __s = 0; __s < __work_group_size; __s++, __slm_bin_hist_summary_offset += __hist_stride)
                {
                    __thread_grf_hist_summary += __utils::__block_load_slm<_HistT, __bin_width>(__slm_bin_hist_summary_offset);
                    __utils::__block_store_slm(__slm_bin_hist_summary_offset, __thread_grf_hist_summary);
                }

                __utils::__block_store_slm(__slm_bin_hist_group_incoming + __local_tid * __bin_width * sizeof(_HistT),
                                     __utils::__scan<_HistT, _HistT>(__thread_grf_hist_summary));
                if (__wg_id != 0)
                    __utils::__block_store<::std::uint32_t, __bin_width>(__p_global_bin_this_group + __local_tid * __bin_width,
                                                           __thread_grf_hist_summary | __hist_updated);
            }
            barrier();
            if (__local_tid == __bin_summary_group_size + 1)
            {
                // this thread to group scan
                simd<_HistT, __bin_count> __grf_hist_summary;
                simd<_HistT, __bin_count + 1> __grf_hist_summary_scan;
                __grf_hist_summary = __utils::__block_load_slm<_HistT, __bin_count>(__slm_bin_hist_group_incoming);
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<__bin_width, 1>(1) =
                    __grf_hist_summary.template select<__bin_width, 1>(0);
#pragma unroll
                for (::std::uint32_t __i = __bin_width; __i < __bin_count; __i += __bin_width)
                {
                    __grf_hist_summary_scan.template select<__bin_width, 1>(__i + 1) =
                        __grf_hist_summary.template select<__bin_width, 1>(__i) + __grf_hist_summary_scan[__i];
                }
                __utils::__block_store_slm<_HistT, __bin_count>(__slm_bin_hist_group_incoming,
                                                        __grf_hist_summary_scan.template select<__bin_count, 1>());
            }
            else if (__local_tid < __bin_summary_group_size)
            {
                // these threads to global sync and update
                simd<_GlobalHistT, __bin_width> __prev_group_hist_sum(0), __prev_group_hist;
                simd_mask<__bin_width> __is_not_accumulated(1);
                do
                {
                    do
                    {
                        __prev_group_hist =
                            lsc_block_load<_GlobalHistT, __bin_width, lsc_data_size::default_size, cache_hint::uncached,
                                           cache_hint::cached>(__p_global_bin_prev_group + __local_tid * __bin_width);
                        fence<fence_mask::sw_barrier>();
                    } while (((__prev_group_hist & __hist_updated) == 0).any() && __wg_id != 0);
                    __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                    __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                    __p_global_bin_prev_group -= __bin_count;
                } while (__is_not_accumulated.any() && __wg_id != 0);
                __prev_group_hist_sum &= __global_offset_mask;
                simd<_GlobalHistT, __bin_width> after_group_hist_sum = __prev_group_hist_sum + __thread_grf_hist_summary;
                __utils::__block_store<::std::uint32_t, __bin_width>(__p_global_bin_this_group + __local_tid * __bin_width,
                                                       after_group_hist_sum | __hist_updated | __global_accumulated);

                __utils::__block_store_slm<::std::uint32_t, __bin_width>(
                    __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobalHistT), __prev_group_hist_sum);
            }
            barrier();
        }
        auto __group_incoming = __utils::__block_load_slm<_HistT, __bin_count>(__slm_bin_hist_group_incoming);
        __global_fix = __utils::__block_load_slm<_GlobalHistT, __bin_count>(__slm_bin_hist_global_incoming) - __group_incoming;
        if (__local_tid > 0)
        {
            __subgroup_offset = __group_incoming + __utils::__block_load_slm<_HistT, __bin_count>((__local_tid - 1) * __hist_stride);
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

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  __data_per_work_item = 256, __bin_count = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change __slm to reuse

        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __bin_count * sizeof(_HistT);
        const ::std::uint32_t __slm_lookup_subgroup = __local_tid * sizeof(_HistT) * __bin_count;

        simd<_HistT, __bin_count> __bin_offset;
        simd<_HistT, __data_per_work_item> __ranks;
        simd<_KeyT, __data_per_work_item> __keys;
        simd<_BinT, __data_per_work_item> __bins;
        simd<_DeviceAddrT, __data_per_step> __lane_id(0, 1);

        const _DeviceAddrT __io_offset = __data_per_work_item * (__wg_id * __wg_size + __local_tid);
        constexpr _KeyT __default_key = __utils::__sort_identity<_KeyT, __is_ascending>();

        __load_keys(__io_offset, __keys, __default_key);

        __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

        __reset_bin_counters(__slm_bin_hist_this_thread);

        fence<fence_mask::local_barrier>();

        __ranks = __rank_slm(__bins, __slm_bin_hist_this_thread, __local_tid);

        barrier();

        simd<_HistT, __bin_count> __subgroup_offset;
        simd<_GlobalHistT, __bin_count> __global_fix;

        _GlobalHistT* __p_global_bin_start_buffer_allstages = reinterpret_cast<_GlobalHistT*>(__p_global_buffer);
        _GlobalHistT* __p_global_bin_start_buffer =
            __p_global_bin_start_buffer_allstages + __bin_count * __stage_count + __bin_count * __wg_count * __stage;

        _GlobalHistT* __p_global_bin_this_group = __p_global_bin_start_buffer + __bin_count * __wg_id;
        _GlobalHistT* __p_global_bin_prev_group = __p_global_bin_start_buffer + __bin_count * (__wg_id - 1);
        __p_global_bin_prev_group = (0 == __wg_id) ? (__p_global_bin_start_buffer_allstages + __bin_count * __stage)
                                               : (__p_global_bin_this_group - __bin_count);

        __update_group_rank(__local_tid, __wg_id, __subgroup_offset, __global_fix, __p_global_bin_prev_group,
                        __p_global_bin_this_group);
        barrier();
        {
            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            _slm_lookup_t<_HistT> __subgroup_lookup(__slm_lookup_subgroup);
            simd<_HistT, __data_per_work_item> __wg_offset =
                __ranks + __subgroup_lookup.template __lookup<__data_per_work_item>(__subgroup_offset, __bins);
            barrier();

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(simd<::std::uint32_t, __data_per_work_item>(__wg_offset) * sizeof(_KeyT),
                                                           __keys);
        }
        barrier();
        _slm_lookup_t<_GlobalHistT> __global_fix_lookup(__reorder_slm_size);
        if (__local_tid == 0)
        {
            __global_fix_lookup.__setup(__global_fix);
        }
        barrier();
        {
            __keys = __utils::__block_load_slm<_KeyT, __data_per_work_item>(__local_tid * __data_per_work_item * sizeof(_KeyT));

            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            simd<_HistT, __data_per_work_item> __group_offset =
                __utils::__create_simd<_HistT, __data_per_work_item>(__local_tid * __data_per_work_item, 1);

            simd<_DeviceAddrT, __data_per_work_item> __global_offset =
                __group_offset + __global_fix_lookup.template __lookup<__data_per_work_item>(__bins);

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__output, __global_offset * sizeof(_KeyT), __keys,
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

template <typename _KeyT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count, ::std::uint32_t __hist_work_group_count,
          ::std::uint32_t __hist_work_group_size, bool __is_ascending, typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <typename _KeyT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count, ::std::uint32_t __hist_work_group_count,
          ::std::uint32_t __hist_work_group_size, bool __is_ascending, typename... _Name>
struct __radix_sort_onesweep_histogram_submitter<
    _KeyT, __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size, __is_ascending,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range, typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _Range&& __rng, const _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size, __hist_work_group_size);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                __cgh.depends_on(__e);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        __global_histogram<_KeyT, decltype(__data), __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size,
                                         __is_ascending>(__nd_item, __n, __data, __global_offset_data);
                    });
            });
    }
};

template <::std::uint32_t __stage_count, ::std::uint32_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t __stage_count, ::std::uint32_t __bin_count, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    __stage_count, __bin_count, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__stage_count * __bin_count, __bin_count);
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

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename... _Name>
struct __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRange, typename _OutRange, typename _TmpData>
    sycl::event
    operator()(sycl::queue& __q, _InRange& __rng, _OutRange& __out_rng, const _TmpData& __tmp_data,
               ::std::uint32_t __sweep_work_group_count, ::std::size_t __n, ::std::uint32_t __stage,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size, __work_group_size);
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng, __out_rng);
                auto __in_data = __rng.data();
                auto __out_data = __out_rng.data();
                __cgh.depends_on(__e);
                __radix_sort_onesweep_slm_reorder_kernel<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
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

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _Range>
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

    using _GlobalHistT = ::std::uint32_t;
    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr ::std::uint32_t __hist_work_group_count = 64;
    constexpr ::std::uint32_t __hist_work_group_size = 64;

    const ::std::uint32_t __sweep_work_group_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __data_per_work_item);
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);

    // Memory for __sync_buffer_size is used by onesweep kernel implicitly
    // TODO: pass a pointer to the the memory allocated with __sync_buffer_size to onesweep kernel
    const ::std::uint32_t __sync_buffer_size = __sweep_work_group_count * __bin_count * __stage_count * sizeof(_GlobalHistT); //bytes
    constexpr ::std::uint32_t __global_offset_size = __bin_count * __stage_count * sizeof(_GlobalHistT);
    ::std::size_t __temp_buffer_size = __global_offset_size + __sync_buffer_size;

    const ::std::size_t __full_buffer_size_global_hist = __temp_buffer_size * sizeof(::std::uint8_t);
    const ::std::size_t __full_buffer_size_output = __n * sizeof(_KeyT);
    const ::std::size_t __full_buffer_size = __full_buffer_size_global_hist + __full_buffer_size_output;

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
        __radix_sort_onesweep_histogram_submitter<_KeyT, __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size,
                                                  __is_ascending, _EsimdRadixSortHistogram>()(__q, __rng, __p_global_offset,
                                                                                            __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _EsimdRadixSortScan>()(__q, __p_global_offset,
                                                                                                __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        if ((__stage % 2) == 0)
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _EsimdRadixSortSweepEven>()(
                __q, __rng, __out_rng, __p_globl_hist_buffer, __sweep_work_group_count, __n, __stage, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _EsimdRadixSortSweepOdd>()(
                __q, __out_rng, __rng, __p_globl_hist_buffer, __sweep_work_group_count, __n, __stage, __event_chain);
        }
    }

    if constexpr (__stage_count % 2 != 0)
    {
        __event_chain =
            __radix_sort_copyback_submitter<_KeyT, _EsimdRadixSortCopyback>()(__q, __out_rng, __rng, __n, __event_chain);
    }

    __event_chain = __q.submit(
        [__event_chain, __p_temp_memory, __q](sycl::handler& __cgh)
        {
            __cgh.depends_on(__event_chain);
            __cgh.host_task([=]() { sycl::free(__p_temp_memory, __q); });
        });

    return __event_chain;
}

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
