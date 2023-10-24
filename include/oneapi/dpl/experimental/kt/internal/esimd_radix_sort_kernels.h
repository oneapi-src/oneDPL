// -*- C++ -*-
//===-- esimd_radix_sort_kernels.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_KERNELS_H

#include <ext/intel/esimd.hpp>
#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>

#include "esimd_radix_sort_utils.h"
#include "../../../pstl/utils.h"

namespace oneapi::dpl::experimental::kt::esimd::__impl
{

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _InputT>
void
__one_wg_kernel(sycl::nd_item<1> __idx, ::std::uint32_t __n, const _InputT& __input)
{
    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint16_t;
    using _DeviceAddrT = ::std::uint32_t;

    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
    constexpr _BinT __mask = __bin_count - 1;
    constexpr ::std::uint32_t __hist_stride = sizeof(_HistT) * __bin_count;

    constexpr ::std::uint32_t __reorder_slm_size = __data_per_work_item * sizeof(_KeyT) * __work_group_size;
    constexpr ::std::uint32_t __bin_hist_slm_size = __hist_stride * __work_group_size;
    constexpr ::std::uint32_t __incoming_offset_slm_size = (__bin_count + 1) * sizeof(_HistT);

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  __data_per_work_item = 256, __bin_count = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    __dpl_esimd_ns::slm_init(::std::max(__reorder_slm_size, __bin_hist_slm_size + __incoming_offset_slm_size));

    const ::std::uint32_t __local_tid = __idx.get_local_linear_id();

    const ::std::uint32_t __slm_reorder_this_thread = __local_tid * __data_per_work_item * sizeof(_KeyT);
    const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __hist_stride;

    __dpl_esimd_ns::simd<_HistT, __bin_count> __bin_offset;
    __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_work_item> __write_addr;
    __dpl_esimd_ns::simd<_KeyT, __data_per_work_item> __keys;
    __dpl_esimd_ns::simd<_BinT, __data_per_work_item> __bins;
    __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_step> __lane_id(0, 1);

    const _DeviceAddrT __io_offset = __data_per_work_item * __local_tid;

    static_assert(__data_per_work_item % __data_per_step == 0);
    static_assert(__bin_count % 128 == 0);
    static_assert(__bin_count % 32 == 0);
#pragma unroll
    for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
    {
        __dpl_esimd_ns::simd_mask<__data_per_step> __m = (__io_offset + __lane_id + __s) < __n;
        __keys.template select<__data_per_step, 1>(__s) =
            __dpl_esimd_ns::merge(__utils::__gather<_KeyT, __data_per_step>(__input, __lane_id, __io_offset + __s, __m),
                  __dpl_esimd_ns::simd<_KeyT, __data_per_step>(__utils::__sort_identity<_KeyT, __is_ascending>()), __m);
    }

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

        __bin_offset = 0;
#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += 1)
        {
            __write_addr[__s] = __bin_offset[__bins[__s]];
            __bin_offset[__bins[__s]] += 1;
        }

        /*
        first write to slm,
        then do column scan by group, each thread to 32c*8r,
        then last row do exclusive scan as incoming offset
        then every thread add local sum with sum of previous group and incoming offset
        */
        {
            __dpl_esimd_ns::barrier();
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
            {
                __utils::__block_store_slm<::std::uint32_t, 64>(
                    __slm_bin_hist_this_thread + __s * sizeof(_HistT),
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>());
            }
            __dpl_esimd_ns::barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            if (__local_tid < __bin_summary_group_size)
            {
                constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;
                constexpr ::std::uint32_t __bin_width_ud = __bin_width * sizeof(_HistT) / sizeof(::std::uint32_t);
                ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * __bin_width * sizeof(_HistT);
                __dpl_esimd_ns::simd<_HistT, __bin_width> ___thread_grf_hist_summary;
                __dpl_esimd_ns::simd<::std::uint32_t, __bin_width_ud> __tmp;

                ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>() =
                    __utils::__block_load_slm<::std::uint32_t, __bin_width_ud>(__slm_bin_hist_summary_offset);
                __slm_bin_hist_summary_offset += __hist_stride;
                for (::std::uint32_t __s = 1; __s < __work_group_size - 1; __s++)
                {
                    __tmp = __utils::__block_load_slm<::std::uint32_t, __bin_width_ud>(__slm_bin_hist_summary_offset);
                    ___thread_grf_hist_summary += __tmp.template bit_cast_view<_HistT>();
                    __utils::__block_store_slm<::std::uint32_t, __bin_width_ud>(
                        __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>());
                    __slm_bin_hist_summary_offset += __hist_stride;
                }
                __tmp = __utils::__block_load_slm<::std::uint32_t, __bin_width_ud>(__slm_bin_hist_summary_offset);
                ___thread_grf_hist_summary += __tmp.template bit_cast_view<_HistT>();
                ___thread_grf_hist_summary = __utils::__scan<_HistT, _HistT>(___thread_grf_hist_summary);
                __utils::__block_store_slm<::std::uint32_t, __bin_width_ud>(
                    __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>());
            }
            __dpl_esimd_ns::barrier();
            if (__local_tid == 0)
            {
                __dpl_esimd_ns::simd<_HistT, __bin_count> __grf_hist_summary;
                __dpl_esimd_ns::simd<_HistT, __bin_count + 1> __grf_hist_summary_scan;
#pragma unroll
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __grf_hist_summary.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>() =
                        __utils::__block_load_slm<::std::uint32_t, 64>((__work_group_size - 1) * __hist_stride + __s * sizeof(_HistT));
                }
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<32, 1>(1) = __grf_hist_summary.template select<32, 1>(0);
#pragma unroll
                for (::std::uint32_t __i = 32; __i < __bin_count; __i += 32)
                {
                    __grf_hist_summary_scan.template select<32, 1>(__i + 1) =
                        __grf_hist_summary.template select<32, 1>(__i) + __grf_hist_summary_scan[__i];
                }
#pragma unroll
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __utils::__block_store_slm<::std::uint32_t, 64>(
                        __bin_hist_slm_size + __s * sizeof(_HistT),
                        __grf_hist_summary_scan.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>());
                }
            }
            __dpl_esimd_ns::barrier();
            {
#pragma unroll
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>() =
                        __utils::__block_load_slm<::std::uint32_t, 64>(__bin_hist_slm_size + __s * sizeof(_HistT));
                }
                if (__local_tid > 0)
                {
#pragma unroll
                    for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                    {
                        __dpl_esimd_ns::simd<_HistT, 128> __group_local_sum;
                        __group_local_sum.template bit_cast_view<::std::uint32_t>() =
                            __utils::__block_load_slm<::std::uint32_t, 64>((__local_tid - 1) * __hist_stride + __s * sizeof(_HistT));
                        __bin_offset.template select<128, 1>(__s) += __group_local_sum;
                    }
                }
            }
            __dpl_esimd_ns::barrier();
        }

#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
        {
            __dpl_esimd_ns::simd<::std::uint16_t, __data_per_step> __bins_uw = __bins.template select<__data_per_step, 1>(__s);
            __write_addr.template select<__data_per_step, 1>(__s) += __bin_offset.template iselect(__bins_uw);
        }

        if (__stage != __stage_count - 1)
        {
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __utils::__vector_store<_KeyT, 1, __data_per_step>(
                    __write_addr.template select<__data_per_step, 1>(__s) * sizeof(_KeyT),
                    __keys.template select<__data_per_step, 1>(__s));
            }
            __dpl_esimd_ns::barrier();
            __keys = __utils::__block_load_slm<_KeyT, __data_per_work_item>(__slm_reorder_this_thread);
        }
    }
#pragma unroll
    for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
    {
        __utils::__scatter<_KeyT, __data_per_step>(__input, __write_addr.template select<__data_per_step, 1>(__s),
                                             __keys.template select<__data_per_step, 1>(__s),
                                             __write_addr.template select<__data_per_step, 1>(__s) < __n);
    }
}

template <typename _KeyT, typename _InputT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count, ::std::uint32_t __hist_work_group_count,
          ::std::uint32_t __hist_work_group_size, bool __is_ascending>
void
__global_histogram(sycl::nd_item<1> __idx, size_t __n, const _InputT& __input, ::std::uint32_t* __p_global_offset)
{
    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint32_t;
    using _GlobalHistT = ::std::uint32_t;

    __dpl_esimd_ns::slm_init(16384);

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

    __dpl_esimd_ns::simd<_KeyT, __hist_data_per_work_item> __keys;
    __dpl_esimd_ns::simd<_BinT, __hist_data_per_work_item> __bins;

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
    __dpl_esimd_ns::barrier();

//#pragma unroll
    for (::std::uint32_t __stage_block = 0; __stage_block < __stage_block_count; ++__stage_block)
    {
        __dpl_esimd_ns::simd<_GlobalHistT, __bin_count * __stages_per_block> __state_hist_grf(0);
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
                __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_offsets(0, 1);
//#pragma unroll
                for (::std::uint32_t __step_offset = 0; __step_offset < __hist_data_per_work_item; __step_offset += __data_per_step)
                {
                    __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __offsets = __lane_offsets + __step_offset + __wi_offset;
                    __dpl_esimd_ns::simd_mask<__data_per_step> __is_in_range = __offsets < __n;
                    __dpl_esimd_ns::simd<_KeyT, __data_per_step> data = __utils::__gather<_KeyT, __data_per_step>(__input, __offsets, 0, __is_in_range);
                    __dpl_esimd_ns::simd<_KeyT, __data_per_step> sort_identities = __utils::__sort_identity<_KeyT, __is_ascending>();
                    __keys.template select<__data_per_step, 1>(__step_offset) = __dpl_esimd_ns::merge(data, sort_identities, __is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
//#pragma unroll
            for (::std::uint32_t __stage_local = 0; __stage_local < __stages_per_block; ++__stage_local)
            {
                constexpr _BinT __mask = __bin_count - 1;
                ::std::uint32_t __stage_global = __stage_block_start + __stage_local;
                __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys),
                                                 __stage_global * __radix_bits);
//#pragma unroll
                for (::std::uint32_t __i = 0; __i < __hist_data_per_work_item; ++__i)
                {
                    ++__state_hist_grf[__stage_local * __bin_count + __bins[__i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
//#pragma unroll
        for (::std::uint32_t __grf_offset = 0; __grf_offset < __bin_count * __stages_per_block; __grf_offset += __data_per_step)
        {
            ::std::uint32_t slm_offset = __stage_block_start * __bin_count + __grf_offset;
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __slm_byte_offsets(slm_offset * sizeof(_GlobalHistT), sizeof(_GlobalHistT));
            __dpl_esimd_ens::lsc_slm_atomic_update<__dpl_esimd_ns::atomic_op::add, _GlobalHistT, __data_per_step>(
                __slm_byte_offsets, __state_hist_grf.template select<__data_per_step, 1>(__grf_offset), 1);
        }
        __dpl_esimd_ns::barrier();
    }

    // 4. Reduce group-local historgrams from SLM into global histograms in global memory
    __dpl_esimd_ns::simd<_GlobalHistT, __group_hist_size> __group_hist =
        __utils::__block_load_slm<_GlobalHistT, __group_hist_size>(__local_id * __group_hist_size *
                                                                    sizeof(_GlobalHistT));
    __dpl_esimd_ns::simd<::std::uint32_t, __group_hist_size> __byte_offsets(0, sizeof(_GlobalHistT));
    __dpl_esimd_ens::lsc_atomic_update<__dpl_esimd_ns::atomic_op::add>(__p_global_offset + __local_id * __group_hist_size,
                                                      __byte_offsets, __group_hist,
                                                      __dpl_esimd_ns::simd_mask<__group_hist_size>(1));
}

template <::std::uint32_t __bit_count>
inline __dpl_esimd_ns::simd<::std::uint32_t, 32>
__match_bins(__dpl_esimd_ns::simd<::std::uint32_t, 32> __bins, ::std::uint32_t __local_tid)
{
    //instruction count is 5*__bit_count, so 40 for 8 bits.
    //performance is about 2 u per __bit for processing size 512 (will call this 16 times)
    // per bits 5 inst * 16 segments * 4 stages = 320 instructions, * 8 threads = 2560, /1.6G = 1.6 us.
    // 8 bits is 12.8 us
    __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
    __dpl_esimd_ns::simd<::std::uint32_t, 32> __matched_bins(0xffffffff);
#pragma unroll
    for (int __i = 0; __i < __bit_count; __i++)
    {
        __dpl_esimd_ns::simd<::std::uint32_t, 32> __bit = (__bins >> __i) & 1;                                         // and
        __dpl_esimd_ns::simd<::std::uint32_t, 32> __x = __dpl_esimd_ns::merge<::std::uint32_t, 32>(0, -1, __bit != 0); // sel
        ::std::uint32_t __ones = __dpl_esimd_ns::pack_mask(__bit != 0);                                                // mov
        __matched_bins = __matched_bins & (__x ^ __ones);                                                              // bfn
    }
    __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
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
    _GlobalHistT* __p_global_hist;
    _GlobalHistT* __p_group_hists;

    __radix_sort_onesweep_slm_reorder_kernel(::std::uint32_t __n, ::std::uint32_t __stage, _InputT __input, _OutputT __output,
                                           _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists)
        : __n(__n), __stage(__stage), __input(__input), __output(__output), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists)
    {
    }

    inline void
    __load_keys(::std::uint32_t __io_offset, __dpl_esimd_ns::simd<_KeyT, __data_per_work_item>& __keys, _KeyT __default_key) const
    {
        static_assert(__data_per_work_item % __data_per_step == 0);

        bool __is_full_block = (__io_offset + __data_per_work_item) < __n;
        if (__is_full_block)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __keys.template select<__data_per_step, 1>(__s) = __utils::__gather<_KeyT, __data_per_step>(__input, __offset, 0);
            }
        }
        else
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __dpl_esimd_ns::simd_mask<__data_per_step> __m = __offset < __n;
                __keys.template select<__data_per_step, 1>(__s) =
                    __dpl_esimd_ns::merge(__utils::__gather<_KeyT, __data_per_step>(__input, __offset, 0, __m),
                                          __dpl_esimd_ns::simd<_KeyT, __data_per_step>(__default_key), __m);
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
        constexpr int __bins_per_step = 32;
        static_assert(__data_per_work_item % __bins_per_step == 0);

        constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
        __dpl_esimd_ns::simd<::std::uint32_t, __data_per_work_item> __ranks;
        __utils::__block_store_slm<_HistT, __bin_count>(__slm_counter_offset, 0);
        __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __remove_right_lanes, __lane_id(0, 1);
        __remove_right_lanes = 0x7fffffff >> (__bins_per_step - 1 - __lane_id);
#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __bins_per_step)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __this_bins = __bins.template select<__bins_per_step, 1>(__s);
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __matched_bins = __match_bins<__radix_bits>(__this_bins, __local_tid); // 40 insts
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __pre_rank, __this_round_rank;
            __pre_rank = __utils::__vector_load<_HistT, 1, __bins_per_step>(__slm_counter_offset +
                                                                 __this_bins * sizeof(_HistT)); // 2 mad+load.__slm
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            __this_round_rank = __dpl_esimd_ns::cbit(__matched_left_lanes);
            auto __this_round_count = __dpl_esimd_ns::cbit(__matched_bins);
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
                    __dpl_esimd_ns::simd<_GlobalHistT, __bin_count>& __global_fix, _GlobalHistT* __p_prev_group_hist,
                    _GlobalHistT* __p_this_group_hist) const
    {
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
            __dpl_esimd_ns::barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;

            static_assert(__bin_count % __bin_width == 0);

            __dpl_esimd_ns::simd<_HistT, __bin_width> __thread_grf_hist_summary(0);
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
                    __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                           __thread_grf_hist_summary | __hist_updated);
            }
            __dpl_esimd_ns::barrier();
            if (__local_tid == __bin_summary_group_size + 1)
            {
                // this thread to group scan
                __dpl_esimd_ns::simd<_HistT, __bin_count> __grf_hist_summary;
                __dpl_esimd_ns::simd<_HistT, __bin_count + 1> __grf_hist_summary_scan;
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
                __dpl_esimd_ns::simd<_GlobalHistT, __bin_width> __prev_group_hist_sum(0), __prev_group_hist;
                __dpl_esimd_ns::simd_mask<__bin_width> __is_not_accumulated(1);
                do
                {
                    do
                    {
                        __prev_group_hist =
                            __dpl_esimd_ens::lsc_block_load<_GlobalHistT, __bin_width, __dpl_esimd_ens::lsc_data_size::default_size, __dpl_esimd_ens::cache_hint::uncached,
                                           __dpl_esimd_ens::cache_hint::cached>(__p_prev_group_hist + __local_tid * __bin_width);
                        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
                    } while (((__prev_group_hist & __hist_updated) == 0).any() && __wg_id != 0);
                    __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                    __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                    __p_prev_group_hist -= __bin_count;
                } while (__is_not_accumulated.any() && __wg_id != 0);
                __prev_group_hist_sum &= __global_offset_mask;
                __dpl_esimd_ns::simd<_GlobalHistT, __bin_width> after_group_hist_sum = __prev_group_hist_sum + __thread_grf_hist_summary;
                __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                       after_group_hist_sum | __hist_updated | __global_accumulated);

                __utils::__block_store_slm<::std::uint32_t, __bin_width>(
                    __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobalHistT), __prev_group_hist_sum);
            }
            __dpl_esimd_ns::barrier();
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
        __dpl_esimd_ns::slm_init(128 * 1024);

        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_id = __idx.get_group(0);
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __wg_count = __idx.get_group_range(0);

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  __data_per_work_item = 256, __bin_count = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change __slm to reuse

        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __bin_count * sizeof(_HistT);
        const ::std::uint32_t __slm_lookup_subgroup = __local_tid * sizeof(_HistT) * __bin_count;

        __dpl_esimd_ns::simd<_HistT, __bin_count> __bin_offset;
        __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __ranks;
        __dpl_esimd_ns::simd<_KeyT, __data_per_work_item> __keys;
        __dpl_esimd_ns::simd<_BinT, __data_per_work_item> __bins;
        __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_step> __lane_id(0, 1);

        const _DeviceAddrT __io_offset = __data_per_work_item * (__wg_id * __wg_size + __local_tid);
        constexpr _KeyT __default_key = __utils::__sort_identity<_KeyT, __is_ascending>();

        __load_keys(__io_offset, __keys, __default_key);

        __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

        __reset_bin_counters(__slm_bin_hist_this_thread);

        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::local_barrier>();

        __ranks = __rank_slm(__bins, __slm_bin_hist_this_thread, __local_tid);

        __dpl_esimd_ns::barrier();

        __dpl_esimd_ns::simd<_HistT, __bin_count> __subgroup_offset;
        __dpl_esimd_ns::simd<_GlobalHistT, __bin_count> __global_fix;

        _GlobalHistT* __p_this_group_hist = __p_group_hists + __bin_count * __wg_id;
        // First group contains global hist to propagate it to other groups during synchronization
        _GlobalHistT* __p_prev_group_hist = (0 == __wg_id)? __p_global_hist : __p_this_group_hist - __bin_count;
        __update_group_rank(__local_tid, __wg_id, __subgroup_offset, __global_fix, __p_prev_group_hist, __p_this_group_hist);
        __dpl_esimd_ns::barrier();
        {
            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            _slm_lookup_t<_HistT> __subgroup_lookup(__slm_lookup_subgroup);
            __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __wg_offset =
                __ranks + __subgroup_lookup.template __lookup<__data_per_work_item>(__subgroup_offset, __bins);
            __dpl_esimd_ns::barrier();

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__dpl_esimd_ns::simd<::std::uint32_t, __data_per_work_item>(__wg_offset) * sizeof(_KeyT),
                                                           __keys);
        }
        __dpl_esimd_ns::barrier();
        _slm_lookup_t<_GlobalHistT> __global_fix_lookup(__reorder_slm_size);
        if (__local_tid == 0)
        {
            __global_fix_lookup.__setup(__global_fix);
        }
        __dpl_esimd_ns::barrier();
        {
            __keys = __utils::__block_load_slm<_KeyT, __data_per_work_item>(__local_tid * __data_per_work_item * sizeof(_KeyT));

            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __group_offset =
                __utils::__create_simd<_HistT, __data_per_work_item>(__local_tid * __data_per_work_item, 1);

            __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_work_item> __global_offset =
                __group_offset + __global_fix_lookup.template __lookup<__data_per_work_item>(__bins);

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__output, __global_offset * sizeof(_KeyT), __keys,
                                                           __global_offset < __n);
        }
    }
};

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _ValT, typename _InKeysT, typename _OutKeysT, typename _InValsT, typename _OutValsT>
struct __radix_sort_onesweep_by_key_slm_reorder_kernel
{
    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint16_t;
    using _GlobalHistT = ::std::uint32_t;
    using _DeviceAddrT = ::std::uint32_t;

    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr _BinT __mask = __bin_count - 1;
    static constexpr ::std::uint32_t __reorder_slm_size = __data_per_work_item * __work_group_size * (sizeof(_KeyT) + sizeof(_ValT));
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
    _InKeysT __in_keys_data;
    _OutKeysT __out_keys_data;
    _InValsT __in_vals_data;
    _OutValsT __out_vals_data;
    _GlobalHistT* __p_global_hist;
    _GlobalHistT* __p_group_hists;

    __radix_sort_onesweep_by_key_slm_reorder_kernel(::std::uint32_t __n, ::std::uint32_t __stage, _InKeysT __in_keys_data, _OutKeysT __out_keys_data, _InValsT __in_vals_data, _OutValsT __out_vals_data,
                                           _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists)
        : __n(__n), __stage(__stage), __in_keys_data(__in_keys_data), __out_keys_data(__out_keys_data), __in_vals_data(__in_vals_data), __out_vals_data(__out_vals_data), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists)
    {
    }

    inline void
    __load_keys(::std::uint32_t __io_offset, __dpl_esimd_ns::simd<_KeyT, __data_per_work_item>& __keys, _KeyT __default_key) const
    {
        static_assert(__data_per_work_item % __data_per_step == 0);

        bool __is_full_block = (__io_offset + __data_per_work_item) < __n;
        if (__is_full_block)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __keys.template select<__data_per_step, 1>(__s) = __utils::__gather<_KeyT, __data_per_step>(__in_keys_data, __offset, 0);
            }
        }
        else
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __dpl_esimd_ns::simd_mask<__data_per_step> __m = __offset < __n;
                __keys.template select<__data_per_step, 1>(__s) =
                    __dpl_esimd_ns::merge(__utils::__gather<_KeyT, __data_per_step>(__in_keys_data, __offset, 0, __m),
                                          __dpl_esimd_ns::simd<_KeyT, __data_per_step>(__default_key), __m);
            }
        }
    }

    inline void
    __load_vals(::std::uint32_t __io_offset, __dpl_esimd_ns::simd<_ValT, __data_per_work_item>& __vals) const
    {
        static_assert(__data_per_work_item % __data_per_step == 0);

        bool __is_full_block = (__io_offset + __data_per_work_item) < __n;
        if (__is_full_block)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __vals.template select<__data_per_step, 1>(__s) = __utils::__gather<_ValT, __data_per_step>(__in_vals_data, __offset, 0);
            }
        }
        else
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_id(0, 1);
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd __offset = __io_offset + __s + __lane_id;
                __dpl_esimd_ns::simd_mask<__data_per_step> __m = __offset < __n;
                __vals.template select<__data_per_step, 1>(__s) = __utils::__gather<_ValT, __data_per_step>(__in_vals_data, __offset, 0, __m);
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
        constexpr int __bins_per_step = 32;
        static_assert(__data_per_work_item % __bins_per_step == 0);

        constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
        __dpl_esimd_ns::simd<::std::uint32_t, __data_per_work_item> __ranks;
        __utils::__block_store_slm<_HistT, __bin_count>(__slm_counter_offset, 0);
        __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __remove_right_lanes, __lane_id(0, 1);
        __remove_right_lanes = 0x7fffffff >> (__bins_per_step - 1 - __lane_id);
#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __bins_per_step)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __this_bins = __bins.template select<__bins_per_step, 1>(__s);
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __matched_bins = __match_bins<__radix_bits>(__this_bins, __local_tid); // 40 insts
            __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step> __pre_rank, __this_round_rank;
            __pre_rank = __utils::__vector_load<_HistT, 1, __bins_per_step>(__slm_counter_offset +
                                                                 __this_bins * sizeof(_HistT)); // 2 mad+load.__slm
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            __this_round_rank = __dpl_esimd_ns::cbit(__matched_left_lanes);
            auto __this_round_count = __dpl_esimd_ns::cbit(__matched_bins);
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
                    __dpl_esimd_ns::simd<_GlobalHistT, __bin_count>& __global_fix, _GlobalHistT* __p_prev_group_hist,
                    _GlobalHistT* __p_this_group_hist) const
    {
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
            __dpl_esimd_ns::barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;

            static_assert(__bin_count % __bin_width == 0);

            __dpl_esimd_ns::simd<_HistT, __bin_width> __thread_grf_hist_summary(0);
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
                    __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                           __thread_grf_hist_summary | __hist_updated);
            }
            __dpl_esimd_ns::barrier();
            if (__local_tid == __bin_summary_group_size + 1)
            {
                // this thread to group scan
                __dpl_esimd_ns::simd<_HistT, __bin_count> __grf_hist_summary;
                __dpl_esimd_ns::simd<_HistT, __bin_count + 1> __grf_hist_summary_scan;
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
                __dpl_esimd_ns::simd<_GlobalHistT, __bin_width> __prev_group_hist_sum(0), __prev_group_hist;
                __dpl_esimd_ns::simd_mask<__bin_width> __is_not_accumulated(1);
                do
                {
                    do
                    {
                        __prev_group_hist =
                            __dpl_esimd_ens::lsc_block_load<_GlobalHistT, __bin_width, __dpl_esimd_ens::lsc_data_size::default_size, __dpl_esimd_ens::cache_hint::uncached,
                                           __dpl_esimd_ens::cache_hint::cached>(__p_prev_group_hist + __local_tid * __bin_width);
                        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
                    } while (((__prev_group_hist & __hist_updated) == 0).any() && __wg_id != 0);
                    __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                    __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                    __p_prev_group_hist -= __bin_count;
                } while (__is_not_accumulated.any() && __wg_id != 0);
                __prev_group_hist_sum &= __global_offset_mask;
                __dpl_esimd_ns::simd<_GlobalHistT, __bin_width> after_group_hist_sum = __prev_group_hist_sum + __thread_grf_hist_summary;
                __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                       after_group_hist_sum | __hist_updated | __global_accumulated);

                __utils::__block_store_slm<::std::uint32_t, __bin_width>(
                    __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobalHistT), __prev_group_hist_sum);
            }
            __dpl_esimd_ns::barrier();
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
        __dpl_esimd_ns::slm_init(128 * 1024);

        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_id = __idx.get_group(0);
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __wg_count = __idx.get_group_range(0);

        // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  __data_per_work_item = 256, __bin_count = 256
        // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
        // change __slm to reuse

        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __bin_count * sizeof(_HistT);
        const ::std::uint32_t __slm_lookup_subgroup = __local_tid * sizeof(_HistT) * __bin_count;

        __dpl_esimd_ns::simd<_HistT, __bin_count> __bin_offset;
        __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __ranks;
        __dpl_esimd_ns::simd<_KeyT, __data_per_work_item> __keys;
        __dpl_esimd_ns::simd<_ValT, __data_per_work_item> __vals;
        __dpl_esimd_ns::simd<_BinT, __data_per_work_item> __bins;
        __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_step> __lane_id(0, 1);

        const _DeviceAddrT __io_offset = __data_per_work_item * (__wg_id * __wg_size + __local_tid);
        constexpr _KeyT __default_key = __utils::__sort_identity<_KeyT, __is_ascending>();

        __load_keys(__io_offset, __keys, __default_key);
        __load_vals(__io_offset, __vals);

        __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

        __reset_bin_counters(__slm_bin_hist_this_thread);

        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::local_barrier>();

        __ranks = __rank_slm(__bins, __slm_bin_hist_this_thread, __local_tid);

        __dpl_esimd_ns::barrier();

        __dpl_esimd_ns::simd<_HistT, __bin_count> __subgroup_offset;
        __dpl_esimd_ns::simd<_GlobalHistT, __bin_count> __global_fix;

        _GlobalHistT* __p_this_group_hist = __p_group_hists + __bin_count * __wg_id;
        // First group contains global hist to propagate it to other groups during synchronization
        _GlobalHistT* __p_prev_group_hist = (0 == __wg_id)? __p_global_hist : __p_this_group_hist - __bin_count;
        __update_group_rank(__local_tid, __wg_id, __subgroup_offset, __global_fix, __p_prev_group_hist, __p_this_group_hist);
        __dpl_esimd_ns::barrier();
        {
            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            _slm_lookup_t<_HistT> __subgroup_lookup(__slm_lookup_subgroup);
            __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __wg_offset =
                __ranks + __subgroup_lookup.template __lookup<__data_per_work_item>(__subgroup_offset, __bins);

            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_work_item> __wg_offset_keys = __wg_offset * sizeof(_KeyT);
            __dpl_esimd_ns::simd<::std::uint32_t, __data_per_work_item> __wg_offset_vals = __wg_size * __data_per_work_item * sizeof(_KeyT) + __wg_offset * sizeof(_ValT);
            __dpl_esimd_ns::barrier();

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__wg_offset_keys, __keys);
            __utils::__vector_store<_ValT, 1, __data_per_work_item>(__wg_offset_vals, __vals);
        }
        __dpl_esimd_ns::barrier();
        _slm_lookup_t<_GlobalHistT> __global_fix_lookup(__reorder_slm_size);
        if (__local_tid == 0)
        {
            __global_fix_lookup.__setup(__global_fix);
        }
        __dpl_esimd_ns::barrier();
        {
            __keys = __utils::__block_load_slm<_KeyT, __data_per_work_item>(__local_tid * __data_per_work_item * sizeof(_KeyT));
            __vals = __utils::__block_load_slm<_ValT, __data_per_work_item>(__wg_size * __data_per_work_item * sizeof(_KeyT) + __local_tid * __data_per_work_item * sizeof(_ValT));

            __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

            __dpl_esimd_ns::simd<_HistT, __data_per_work_item> __group_offset =
                __utils::__create_simd<_HistT, __data_per_work_item>(__local_tid * __data_per_work_item, 1);

            __dpl_esimd_ns::simd<_DeviceAddrT, __data_per_work_item> __global_offset =
                __group_offset + __global_fix_lookup.template __lookup<__data_per_work_item>(__bins);

            __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__out_keys_data, __global_offset * sizeof(_KeyT), __keys,
                                                           __global_offset < __n);
            __utils::__vector_store<_ValT, 1, __data_per_work_item>(__out_vals_data, __global_offset * sizeof(_ValT), __vals,
                                                           __global_offset < __n);
        }
    }
};

} // oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_KERNELS_H
