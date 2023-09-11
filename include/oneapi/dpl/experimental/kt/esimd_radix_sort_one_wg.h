// -*- C++ -*-
//===-- esimd_radix_sort_one_wg.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_ONE_WG_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_ONE_WG_H

#include <ext/intel/esimd.hpp>
#include "../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "esimd_radix_sort_utils.h"
#include "../../pstl/utils.h"

#include <cstdint>
#include <cassert>

namespace oneapi::dpl::experimental::kt::esimd::__impl
{
template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _InputT>
void
__one_wg_kernel(sycl::nd_item<1> __idx, ::std::uint32_t __n, const _InputT& __input)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

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
    slm_init(std::max(__reorder_slm_size, __bin_hist_slm_size + __incoming_offset_slm_size));

    const ::std::uint32_t __local_tid = __idx.get_local_linear_id();

    const ::std::uint32_t __slm_reorder_this_thread = __local_tid * __data_per_work_item * sizeof(_KeyT);
    const ::std::uint32_t __slm_bin_hist_this_thread = __local_tid * __hist_stride;

    simd<_HistT, __bin_count> __bin_offset;
    simd<_DeviceAddrT, __data_per_work_item> __write_addr;
    simd<_KeyT, __data_per_work_item> __keys;
    simd<_BinT, __data_per_work_item> __bins;
    simd<_DeviceAddrT, __data_per_step> __lane_id(0, 1);

    const _DeviceAddrT __io_offset = __data_per_work_item * __local_tid;

    static_assert(__data_per_work_item % __data_per_step == 0);
    static_assert(__bin_count % 128 == 0);
    static_assert(__bin_count % 32 == 0);
#pragma unroll
    for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
    {
        simd_mask<__data_per_step> __m = (__io_offset + __lane_id + __s) < __n;
        __keys.template select<__data_per_step, 1>(__s) =
            merge(__utils::__gather<_KeyT, __data_per_step>(__input, __lane_id, __io_offset + __s, __m),
                  simd<_KeyT, __data_per_step>(__utils::__sort_identity<_KeyT, __is_ascending>()), __m);
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
            barrier();
#pragma unroll
            for (::std::uint32_t __s = 0; __s < __bin_count; __s += 128)
            {
                __utils::__block_store_slm<::std::uint32_t, 64>(
                    __slm_bin_hist_this_thread + __s * sizeof(_HistT),
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<::std::uint32_t>());
            }
            barrier();
            constexpr ::std::uint32_t __bin_summary_group_size = 8;
            if (__local_tid < __bin_summary_group_size)
            {
                constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;
                constexpr ::std::uint32_t _BIN_WIDTH_UD = __bin_width * sizeof(_HistT) / sizeof(::std::uint32_t);
                ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * __bin_width * sizeof(_HistT);
                simd<_HistT, __bin_width> ___thread_grf_hist_summary;
                simd<::std::uint32_t, _BIN_WIDTH_UD> __tmp;

                ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>() =
                    __utils::__block_load_slm<::std::uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                __slm_bin_hist_summary_offset += __hist_stride;
                for (::std::uint32_t __s = 1; __s < __work_group_size - 1; __s++)
                {
                    __tmp = __utils::__block_load_slm<::std::uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                    ___thread_grf_hist_summary += __tmp.template bit_cast_view<_HistT>();
                    __utils::__block_store_slm<::std::uint32_t, _BIN_WIDTH_UD>(
                        __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>());
                    __slm_bin_hist_summary_offset += __hist_stride;
                }
                __tmp = __utils::__block_load_slm<::std::uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                ___thread_grf_hist_summary += __tmp.template bit_cast_view<_HistT>();
                ___thread_grf_hist_summary = __utils::__scan<_HistT, _HistT>(___thread_grf_hist_summary);
                __utils::__block_store_slm<::std::uint32_t, _BIN_WIDTH_UD>(
                    __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<::std::uint32_t>());
            }
            barrier();
            if (__local_tid == 0)
            {
                simd<_HistT, __bin_count> __grf_hist_summary;
                simd<_HistT, __bin_count + 1> __grf_hist_summary_scan;
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
            barrier();
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
                        simd<_HistT, 128> __group_local_sum;
                        __group_local_sum.template bit_cast_view<::std::uint32_t>() =
                            __utils::__block_load_slm<::std::uint32_t, 64>((__local_tid - 1) * __hist_stride + __s * sizeof(_HistT));
                        __bin_offset.template select<128, 1>(__s) += __group_local_sum;
                    }
                }
            }
            barrier();
        }

#pragma unroll
        for (::std::uint32_t __s = 0; __s < __data_per_work_item; __s += __data_per_step)
        {
            simd<::std::uint16_t, __data_per_step> __bins_uw = __bins.template select<__data_per_step, 1>(__s);
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
            barrier();
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

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_one_wg;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _KernelName>
struct __radix_sort_one_wg_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename... _Name>
struct __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range>
    sycl::event
    operator()(sycl::queue __q, _Range&& __rng, ::std::size_t __n) const
    {
        sycl::nd_range<1> __nd_range{__work_group_size, __work_group_size};
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        __one_wg_kernel<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT>(__nd_item, __n,
                                                                                                         __data);
                    });
            });
    }
};

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _Range>
sycl::event
__one_wg(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;
    using _EsimRadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__esimd_radix_sort_one_wg<_KernelName>>;

    return __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                         _EsimRadixSortKernel>()(__q, ::std::forward<_Range>(__rng), __n);
}

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONE_WG_H
