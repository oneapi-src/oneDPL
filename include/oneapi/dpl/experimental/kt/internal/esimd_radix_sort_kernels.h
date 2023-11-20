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

#include "oneapi/dpl/pstl/onedpl_config.h"

#include <ext/intel/esimd.hpp>
#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>
#include <type_traits>

#include "esimd_radix_sort_utils.h"
#include "../../../pstl/utils.h"

namespace oneapi::dpl::experimental::kt::esimd::__impl
{

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _RngPack>
_ONEDPL_ESIMD_INLINE void
__one_wg_kernel(sycl::nd_item<1> __idx, ::std::uint32_t __n, _RngPack&& __rng_pack)
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
    __dpl_esimd_ns::slm_init<::std::max(__reorder_slm_size, __bin_hist_slm_size + __incoming_offset_slm_size)>();

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
    _ONEDPL_PRAGMA_UNROLL
    for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
    {
        __dpl_esimd_ns::simd_mask<__data_per_step> __m = (__io_offset + __lane_id + __s) < __n;
        __keys.template select<__data_per_step, 1>(__s) =
            __dpl_esimd_ns::merge(__utils::__gather<_KeyT, __data_per_step>(__rng_pack.__keys_acc(), __lane_id, __io_offset + __s, __m),
                  __dpl_esimd_ns::simd<_KeyT, __data_per_step>(__utils::__sort_identity<_KeyT, __is_ascending>()), __m);
    }

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys), __stage * __radix_bits);

        __bin_offset = 0;
        _ONEDPL_PRAGMA_UNROLL
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
            _ONEDPL_PRAGMA_UNROLL
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
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __grf_hist_summary.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>() =
                        __utils::__block_load_slm<::std::uint32_t, 64>((__work_group_size - 1) * __hist_stride + __s * sizeof(_HistT));
                }
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<32, 1>(1) = __grf_hist_summary.template select<32, 1>(0);
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __i = 32; __i < __bin_count; __i += 32)
                {
                    __grf_hist_summary_scan.template select<32, 1>(__i + 1) =
                        __grf_hist_summary.template select<32, 1>(__i) + __grf_hist_summary_scan[__i];
                }
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __utils::__block_store_slm<::std::uint32_t, 64>(
                        __bin_hist_slm_size + __s * sizeof(_HistT),
                        __grf_hist_summary_scan.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>());
                }
            }
            __dpl_esimd_ns::barrier();
            {
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
                {
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>() =
                        __utils::__block_load_slm<::std::uint32_t, 64>(__bin_hist_slm_size + __s * sizeof(_HistT));
                }
                if (__local_tid > 0)
                {
                    _ONEDPL_PRAGMA_UNROLL
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

        _ONEDPL_PRAGMA_UNROLL
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
        {
            __dpl_esimd_ns::simd<::std::uint16_t, __data_per_step> __bins_uw = __bins.template select<__data_per_step, 1>(__s);
            __write_addr.template select<__data_per_step, 1>(__s) += __bin_offset.template iselect(__bins_uw);
        }

        if (__stage != __stage_count - 1)
        {
            _ONEDPL_PRAGMA_UNROLL
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
    _ONEDPL_PRAGMA_UNROLL
    for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
    {
        __utils::__scatter<_KeyT, __data_per_step>(__rng_pack.__keys_acc(), __write_addr.template select<__data_per_step, 1>(__s),
                                             __keys.template select<__data_per_step, 1>(__s),
                                             __write_addr.template select<__data_per_step, 1>(__s) < __n);
    }
}

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint32_t __hist_work_group_count,
          ::std::uint16_t __hist_work_group_size, typename _KeysRng>
_ONEDPL_ESIMD_INLINE void
__global_histogram(sycl::nd_item<1> __idx, size_t __n, const _KeysRng& __keys_rng, ::std::uint32_t* __p_global_offset)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;
    using _BinT = ::std::uint16_t;
    using _HistT = ::std::uint32_t;
    using _GlobalHistT = ::std::uint32_t;

    __dpl_esimd_ns::slm_init<16384>();

    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
    constexpr ::std::uint32_t __hist_data_per_work_item = 128;
    constexpr ::std::uint32_t __device_wide_step = __hist_work_group_count * __hist_work_group_size * __hist_data_per_work_item;

    // Cap the number of histograms to reduce in SLM per __keys_rng range read pass
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

    _ONEDPL_PRAGMA_UNROLL
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
                __utils::__copy_from(__keys_rng.data(), __wi_offset, __keys);
            }
            else
            {
                __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __lane_offsets(0, 1);
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __step_offset = 0; __step_offset < __hist_data_per_work_item; __step_offset += __data_per_step)
                {
                    __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __offsets = __lane_offsets + __step_offset + __wi_offset;
                    __dpl_esimd_ns::simd_mask<__data_per_step> __is_in_range = __offsets < __n;
                    __dpl_esimd_ns::simd<_KeyT, __data_per_step> data = __utils::__gather<_KeyT, __data_per_step>(__keys_rng.data(), __offsets, 0, __is_in_range);
                    __dpl_esimd_ns::simd<_KeyT, __data_per_step> sort_identities = __utils::__sort_identity<_KeyT, __is_ascending>();
                    __keys.template select<__data_per_step, 1>(__step_offset) = __dpl_esimd_ns::merge(data, sort_identities, __is_in_range);
                }
            }
            // 2. Calculate thread-local histogram in GRF
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __stage_local = 0; __stage_local < __stages_per_block; ++__stage_local)
            {
                constexpr _BinT __mask = __bin_count - 1;
                ::std::uint32_t __stage_global = __stage_block_start + __stage_local;
                __bins = __utils::__get_bucket<__mask>(__utils::__order_preserving_cast<__is_ascending>(__keys),
                                                 __stage_global * __radix_bits);
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __i = 0; __i < __hist_data_per_work_item; ++__i)
                {
                    ++__state_hist_grf[__stage_local * __bin_count + __bins[__i]];
                }
            }
        }

        // 3. Reduce thread-local histograms from GRF into group-local histograms in SLM
        _ONEDPL_PRAGMA_UNROLL
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

template <bool __is_ascending, ::std::uint8_t __radix_bits,
          ::std::uint16_t __data_per_work_item, ::std::uint16_t __work_group_size,
          typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel
{
    using _LocOffsetT = ::std::uint16_t;
    using _GlobOffsetT = ::std::uint32_t;

    using _KeyT = typename _InRngPack::_KeyT;
    using _ValT = typename _InRngPack::_ValT;
    static constexpr bool __has_values = !::std::is_void_v<_ValT>;

    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    template<typename _T, ::std::uint16_t _N>
    using _SimdT =  __dpl_esimd_ns::simd<_T, _N>;

    using _LocOffsetSimdT = _SimdT<_LocOffsetT, __data_per_work_item>;
    using _GlobOffsetSimdT = _SimdT<_GlobOffsetT, __data_per_work_item>;
    using _LocHistT = _SimdT<_LocOffsetT, __bin_count>;
    using _GlobHistT = _SimdT<_GlobOffsetT, __bin_count>;

    static constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr _LocOffsetT __mask = __bin_count - 1;
    static constexpr ::std::uint32_t __hist_stride = __bin_count * sizeof(_LocOffsetT);
    static constexpr ::std::uint32_t __work_item_all_hists_size = __work_group_size * __hist_stride;
    static constexpr ::std::uint32_t __group_hist_size = __hist_stride;
    static constexpr ::std::uint32_t __global_hist_size = __bin_count * sizeof(_GlobOffsetT);

    static constexpr ::std::uint32_t __calc_reorder_slm_size()
    {
        if constexpr (__has_values)
            return __work_group_size * __data_per_work_item * (sizeof(_KeyT) + sizeof(_ValT));
        else
            return __work_group_size * __data_per_work_item * sizeof(_KeyT);
    }

    static constexpr ::std::uint32_t __calc_slm_alloc()
    {
        // SLM usage:
        // 1. Getting offsets:
        //      1.1 Scan histograms for each work-item: __work_item_all_hists_size
        //      1.2 Scan group histogram: __group_hist_size
        //      1.3 Accumulate group histogram from previous groups: __global_hist_size
        // 2. Reorder keys in SLM:
        //      2.1 Place global offsets into SLM for lookup: __global_hist_size
        //      2.2 Reorder key-value pairs: __reorder_size
        constexpr ::std::uint32_t __reorder_size = __calc_reorder_slm_size();
        constexpr ::std::uint32_t __offset_calc_substage_slm =
            __work_item_all_hists_size + __group_hist_size + __global_hist_size;
        constexpr ::std::uint32_t __reorder_substage_slm = __reorder_size + __global_hist_size;

        constexpr ::std::uint32_t __slm_size = ::std::max(__offset_calc_substage_slm, __reorder_substage_slm);
        // Workaround: Align SLM allocation at 2048 byte border to avoid internal compiler error.
        // The error happens when allocating 65 * 1024 bytes, when e.g. T=int, DataPerWorkItem=256, WorkGroupSize=64
        // TODO: use __slm_size once the issue with SLM allocation has been fixed
        return oneapi::dpl::__internal::__dpl_ceiling_div(__slm_size, 2048) * 2048;
    }

    const ::std::uint32_t __n;
    const ::std::uint32_t __stage;
    _GlobOffsetT* __p_global_hist;
    _GlobOffsetT* __p_group_hists;
    _InRngPack __in_pack;
    _OutRngPack __out_pack;

    __radix_sort_onesweep_kernel(::std::uint32_t __n, ::std::uint32_t __stage,
                                 _GlobOffsetT* __p_global_hist, _GlobOffsetT* __p_group_hists,
                                 const _InRngPack& __in_pack, const _OutRngPack& __out_pack):
        __n(__n), __stage(__stage),
        __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists),
        __in_pack(__in_pack), __out_pack(__out_pack) {}

    template<typename _SimdPack>
    inline auto
    __load_simd_pack(_SimdPack& __pack, ::std::uint32_t __wg_id, ::std::uint32_t __wg_size, ::std::uint32_t __lid) const
    {
        const _GlobOffsetT __offset = __data_per_work_item * (__wg_id * __wg_size + __lid);
        __load_simd</*__sort_identity_residual=*/true>(__pack.__keys, __in_pack.__keys_acc(), __offset);
        if constexpr (__has_values)
        {
            __load_simd</*__sort_identity_residual=*/false>(__pack.__vals, __in_pack.__vals_acc(), __offset);
        }
    }

    template<bool __sort_identity_residual, typename _T, typename _InSeq>
    inline void
    __load_simd(_SimdT<_T, __data_per_work_item>& __simd, const _InSeq& __in_seq, _GlobOffsetT __glob_offset) const
    {
        static_assert(__data_per_work_item % __data_per_step == 0);

        __dpl_esimd_ns::simd<::std::uint16_t, __data_per_step> __lane_id(0, 1);
        bool __is_full_block = (__glob_offset + __data_per_work_item) < __n;
        if (__is_full_block)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __offset = __glob_offset + __s + __lane_id;
                __simd.template select<__data_per_step, 1>(__s) =
                    __utils::__gather<_T, __data_per_step>(__in_seq, __offset, 0);
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
            {
                __dpl_esimd_ns::simd<::std::uint32_t, __data_per_step> __offset = __glob_offset + __s + __lane_id;
                __dpl_esimd_ns::simd_mask<__data_per_step> __m = __offset < __n;
                auto __gathered = __utils::__gather<_T, __data_per_step>(__in_seq, __offset, 0, __m);
                if constexpr (__sort_identity_residual)
                {
                    constexpr _T __default_item = __utils::__sort_identity<_T, __is_ascending>();
                    auto __default = __dpl_esimd_ns::simd<_T, __data_per_step>(__default_item);
                    __simd.template select<__data_per_step, 1>(__s) =__dpl_esimd_ns::merge(__gathered, __default, __m);
                }
                else
                {
                    __simd.template select<__data_per_step, 1>(__s) = __gathered;
                }
            }
        }
    }

    static inline __dpl_esimd_ns::simd<::std::uint32_t, 32>
    __match_bins(const __dpl_esimd_ns::simd<::std::uint32_t, 32>& __bins, ::std::uint32_t __local_tid)
    {
        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
        __dpl_esimd_ns::simd<::std::uint32_t, 32> __matched_bins(0xffffffff);
        _ONEDPL_PRAGMA_UNROLL
        for (int __i = 0; __i < __radix_bits; __i++)
        {
            __dpl_esimd_ns::simd<::std::uint32_t, 32> __bit = (__bins >> __i) & 1;
            __dpl_esimd_ns::simd<::std::uint32_t, 32> __x =
                __dpl_esimd_ns::merge<::std::uint32_t, 32>(0, -1, __bit != 0);
            ::std::uint32_t __ones = __dpl_esimd_ns::pack_mask(__bit != 0);
            __matched_bins = __matched_bins & (__x ^ __ones);
        }
        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
        return __matched_bins;
    }

    inline auto
    __rank_local(_LocOffsetSimdT& __ranks, _LocOffsetSimdT& __bins,
                 ::std::uint32_t __slm_counter_offset, ::std::uint32_t __local_tid) const
    {
        constexpr int __bins_per_step = 32;
        using _ScanSimdT = __dpl_esimd_ns::simd<::std::uint32_t, __bins_per_step>;

        __utils::__block_store_slm<_LocOffsetT, __bin_count>(__slm_counter_offset, 0);
        _ScanSimdT __remove_right_lanes, __lane_id(0, 1);
        __remove_right_lanes = 0x7fffffff >> (__bins_per_step - 1 - __lane_id);

        static_assert(__data_per_work_item % __bins_per_step == 0);
        _ONEDPL_PRAGMA_UNROLL
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __bins_per_step)
        {
            _ScanSimdT __this_bins = __bins.template select<__bins_per_step, 1>(__s);
            _ScanSimdT __matched_bins = __match_bins(__this_bins, __local_tid);
            _ScanSimdT __pre_rank = __utils::__vector_load<_LocOffsetT, 1, __bins_per_step>(
                __slm_counter_offset + __this_bins * sizeof(_LocOffsetT));
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            _ScanSimdT __this_round_rank = __dpl_esimd_ns::cbit(__matched_left_lanes);
            auto __this_round_count = __dpl_esimd_ns::cbit(__matched_bins);
            auto __rank_after = __pre_rank + __this_round_rank;
            auto __is_leader = __this_round_rank == __this_round_count - 1;
            __utils::__vector_store<_LocOffsetT, 1, __bins_per_step>(
                __slm_counter_offset + __this_bins * sizeof(_LocOffsetT), __rank_after + 1, __is_leader);
            __ranks.template select<__bins_per_step, 1>(__s) = __rank_after;
        }
        __dpl_esimd_ns::barrier();
    }

    inline void
    __rank_global(_LocHistT& __subgroup_offset, _GlobHistT& __global_fix,
                  ::std::uint32_t __local_tid, ::std::uint32_t __wg_id) const
    {
        /*
        first do column scan by group, each thread do 32c,
        then last row do exclusive scan as group incoming __offset
        then every thread add local sum with sum of previous group and incoming __offset
        */
        const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __hist_stride;
        const ::std::uint32_t __slm_bin_hist_group_incoming = __work_group_size * __hist_stride;
        const ::std::uint32_t __slm_bin_hist_global_incoming = __slm_bin_hist_group_incoming + __hist_stride;
        constexpr ::std::uint32_t __global_accumulated = 0x40000000;
        constexpr ::std::uint32_t __hist_updated = 0x80000000;
        constexpr ::std::uint32_t __global_offset_mask = 0x3fffffff;

        _GlobOffsetT* __p_this_group_hist = __p_group_hists + __bin_count * __wg_id;
        // First group contains global histogram to propagate it to other groups during synchronization
        _GlobOffsetT* __p_prev_group_hist = (0 == __wg_id)? __p_global_hist : __p_this_group_hist - __bin_count;

        {
            __dpl_esimd_ns::barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;

            static_assert(__bin_count % __bin_width == 0);

            __dpl_esimd_ns::simd<_LocOffsetT, __bin_width> __thread_grf_hist_summary(0);
            if (__local_tid < __bin_summary_group_size)
            {
                ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * __bin_width * sizeof(_LocOffsetT);
                for (::std::uint32_t __s = 0; __s < __work_group_size; __s++, __slm_bin_hist_summary_offset += __hist_stride)
                {
                    __thread_grf_hist_summary += __utils::__block_load_slm<_LocOffsetT, __bin_width>(__slm_bin_hist_summary_offset);
                    __utils::__block_store_slm(__slm_bin_hist_summary_offset, __thread_grf_hist_summary);
                }

                __utils::__block_store_slm(__slm_bin_hist_group_incoming + __local_tid * __bin_width * sizeof(_LocOffsetT),
                                     __utils::__scan<_LocOffsetT, _LocOffsetT>(__thread_grf_hist_summary));
                if (__wg_id != 0)
                    __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                           __thread_grf_hist_summary | __hist_updated);
            }
            __dpl_esimd_ns::barrier();
            if (__local_tid == __bin_summary_group_size + 1)
            {
                // this thread to group scan
                __dpl_esimd_ns::simd<_LocOffsetT, __bin_count> __grf_hist_summary;
                __dpl_esimd_ns::simd<_LocOffsetT, __bin_count + 1> __grf_hist_summary_scan;
                __grf_hist_summary = __utils::__block_load_slm<_LocOffsetT, __bin_count>(__slm_bin_hist_group_incoming);
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<__bin_width, 1>(1) =
                    __grf_hist_summary.template select<__bin_width, 1>(0);
                _ONEDPL_PRAGMA_UNROLL
                for (::std::uint32_t __i = __bin_width; __i < __bin_count; __i += __bin_width)
                {
                    __grf_hist_summary_scan.template select<__bin_width, 1>(__i + 1) =
                        __grf_hist_summary.template select<__bin_width, 1>(__i) + __grf_hist_summary_scan[__i];
                }
                __utils::__block_store_slm<_LocOffsetT, __bin_count>(__slm_bin_hist_group_incoming,
                                                        __grf_hist_summary_scan.template select<__bin_count, 1>());
            }
            else if (__local_tid < __bin_summary_group_size)
            {
                // these threads to global sync and update
                __dpl_esimd_ns::simd<_GlobOffsetT, __bin_width> __prev_group_hist_sum(0), __prev_group_hist;
                __dpl_esimd_ns::simd_mask<__bin_width> __is_not_accumulated(1);
                do
                {
                    do
                    {
                        __prev_group_hist =
                            __dpl_esimd_ens::lsc_block_load<_GlobOffsetT, __bin_width, __dpl_esimd_ens::lsc_data_size::default_size, __dpl_esimd_ens::cache_hint::uncached,
                                           __dpl_esimd_ens::cache_hint::cached>(__p_prev_group_hist + __local_tid * __bin_width);
                        __dpl_esimd_ns::fence<__dpl_esimd_ns::fence_mask::sw_barrier>();
                    } while (((__prev_group_hist & __hist_updated) == 0).any() && __wg_id != 0);
                    __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                    __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                    __p_prev_group_hist -= __bin_count;
                } while (__is_not_accumulated.any() && __wg_id != 0);
                __prev_group_hist_sum &= __global_offset_mask;
                __dpl_esimd_ns::simd<_GlobOffsetT, __bin_width> after_group_hist_sum = __prev_group_hist_sum + __thread_grf_hist_summary;
                __utils::__block_store<::std::uint32_t, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                       after_group_hist_sum | __hist_updated | __global_accumulated);

                __utils::__block_store_slm<::std::uint32_t, __bin_width>(
                    __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobOffsetT), __prev_group_hist_sum);
            }
            __dpl_esimd_ns::barrier();
        }
        auto __group_incoming = __utils::__block_load_slm<_LocOffsetT, __bin_count>(__slm_bin_hist_group_incoming);
        __global_fix = __utils::__block_load_slm<_GlobOffsetT, __bin_count>(__slm_bin_hist_global_incoming) - __group_incoming;
        if (__local_tid > 0)
        {
            __subgroup_offset = __group_incoming + __utils::__block_load_slm<_LocOffsetT, __bin_count>((__local_tid - 1) * __hist_stride);
        }
        else
            __subgroup_offset = __group_incoming;
        __dpl_esimd_ns::barrier();
    }

    template<typename _SimdPack>
    void inline
    __reorder_reg_to_slm(const _SimdPack& __pack, const _LocOffsetSimdT& __ranks,
                         const _LocOffsetSimdT& __bins, const _LocHistT& __subgroup_offset,
                         ::std::uint32_t __wg_size, ::std::uint32_t __thread_slm_offset) const
    {
        __utils::__slm_lookup<_LocOffsetT> __subgroup_lookup(__thread_slm_offset);
        _LocOffsetSimdT __wg_offset =
            __ranks + __subgroup_lookup.template __lookup<__data_per_work_item>(__subgroup_offset, __bins);
        __dpl_esimd_ns::barrier();

        _GlobOffsetSimdT __wg_offset_keys = __wg_offset * sizeof(_KeyT);
        __utils::__vector_store<_KeyT, 1, __data_per_work_item>(__wg_offset_keys, __pack.__keys);
        if constexpr (__has_values)
        {
            _GlobOffsetSimdT __wg_offset_vals = __wg_size * __data_per_work_item * sizeof(_KeyT) +
                                                __wg_offset * sizeof(_ValT);
            __utils::__vector_store<_ValT, 1, __data_per_work_item>(__wg_offset_vals, __pack.__vals);
        }
        __dpl_esimd_ns::barrier();
    }

    void inline
    __reorder_slm_to_glob(const _GlobHistT& __global_fix, ::std::uint32_t __local_tid, ::std::uint32_t __wg_size) const
    {
        auto __pack = __utils::__make_simd_pack<__data_per_work_item, _KeyT, _ValT>();

        __utils::__slm_lookup<_GlobOffsetT> __global_fix_lookup(__calc_reorder_slm_size());
        if (__local_tid == 0)
            __global_fix_lookup.__setup(__global_fix);
        __dpl_esimd_ns::barrier();

        ::std::uint32_t __keys_slm_offset = __local_tid * __data_per_work_item * sizeof(_KeyT);
        __pack.__keys = __utils::__block_load_slm<_KeyT, __data_per_work_item>(__keys_slm_offset);
        if constexpr (__has_values)
        {
            ::std::uint32_t __vals_slm_offset = __wg_size * __data_per_work_item * sizeof(_KeyT) +
                                                __local_tid * __data_per_work_item * sizeof(_ValT);
            __pack.__vals = __utils::__block_load_slm<_ValT, __data_per_work_item>(__vals_slm_offset);
        }
        const auto __ordered = __utils::__order_preserving_cast<__is_ascending>(__pack.__keys);
        _LocOffsetSimdT __bins = __utils::__get_bucket<__mask>(__ordered, __stage * __radix_bits);

        _LocOffsetSimdT __group_offset =
            __utils::__create_simd<_LocOffsetT, __data_per_work_item>(__local_tid * __data_per_work_item, 1);

        _GlobOffsetSimdT __global_offset =
            __group_offset + __global_fix_lookup.template __lookup<__data_per_work_item>(__bins);

        __utils::__vector_store<_KeyT, 1, __data_per_work_item>(
            __out_pack.__keys_acc(), __global_offset * sizeof(_KeyT), __pack.__keys, __global_offset < __n);
        if constexpr (__has_values)
        {
            __utils::__vector_store<_ValT, 1, __data_per_work_item>(
                __out_pack.__vals_acc(), __global_offset * sizeof(_ValT), __pack.__vals, __global_offset < __n);
        }
    }

    _ONEDPL_ESIMD_INLINE void
    operator()(sycl::nd_item<1> __idx) const SYCL_ESIMD_KERNEL
    {
        __dpl_esimd_ns::slm_init<__calc_slm_alloc()>();

        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_id = __idx.get_group(0);
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __thread_slm_offset = __local_tid * __bin_count * sizeof(_LocOffsetT);

        auto __values_simd_pack = __utils::__make_simd_pack<__data_per_work_item, _KeyT, _ValT>();
        _LocOffsetSimdT __bins;
        _LocOffsetSimdT __ranks;
        _LocHistT __subgroup_offset;
        _GlobHistT __global_fix;

        __load_simd_pack(__values_simd_pack, __wg_id, __wg_size, __local_tid);

        const auto __ordered = __utils::__order_preserving_cast<__is_ascending>(__values_simd_pack.__keys);
        __bins = __utils::__get_bucket<__mask>(__ordered, __stage * __radix_bits);

        __rank_local(__ranks, __bins, __thread_slm_offset, __local_tid);
        __rank_global(__subgroup_offset, __global_fix, __local_tid, __wg_id);

        __reorder_reg_to_slm(__values_simd_pack, __ranks, __bins, __subgroup_offset, __wg_size, __thread_slm_offset);
        __reorder_slm_to_glob(__global_fix, __local_tid, __wg_size);
    }
};

} // oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_KERNELS_H
