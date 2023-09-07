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
template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename InputT>
void
__one_wg_kernel(sycl::nd_item<1> __idx, uint32_t __n, const InputT& __input)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

    using _bin_t = uint16_t;
    using _hist_t = uint16_t;
    using _device_addr_t = uint32_t;

    constexpr uint32_t _BIN_COUNT = 1 << _RadixBits;
    constexpr uint32_t _NBITS = sizeof(_KeyT) * 8;
    constexpr uint32_t _STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(_NBITS, _RadixBits);
    constexpr _bin_t _MASK = _BIN_COUNT - 1;
    constexpr uint32_t _HIST_STRIDE = sizeof(_hist_t) * _BIN_COUNT;

    constexpr uint32_t _REORDER_SLM_SIZE = _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize;
    constexpr uint32_t _BIN_HIST_SLM_SIZE = _HIST_STRIDE * _WorkGroupSize;
    constexpr uint32_t _INCOMING_OFFSET_SLM_SIZE = (_BIN_COUNT + 1) * sizeof(_hist_t);

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  _DataPerWorkItem = 256, _BIN_COUNT = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    slm_init(std::max(_REORDER_SLM_SIZE, _BIN_HIST_SLM_SIZE + _INCOMING_OFFSET_SLM_SIZE));

    const uint32_t __local_tid = __idx.get_local_linear_id();

    const uint32_t __slm_reorder_this_thread = __local_tid * _DataPerWorkItem * sizeof(_KeyT);
    const uint32_t __slm_bin_hist_this_thread = __local_tid * _HIST_STRIDE;

    simd<_hist_t, _BIN_COUNT> __bin_offset;
    simd<_device_addr_t, _DataPerWorkItem> __write_addr;
    simd<_KeyT, _DataPerWorkItem> __keys;
    simd<_bin_t, _DataPerWorkItem> __bins;
    simd<_device_addr_t, _DATA_PER_STEP> __lane_id(0, 1);

    const _device_addr_t __io_offset = _DataPerWorkItem * __local_tid;

    static_assert(_DataPerWorkItem % _DATA_PER_STEP == 0);
    static_assert(_BIN_COUNT % 128 == 0);
    static_assert(_BIN_COUNT % 32 == 0);
#pragma unroll
    for (uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
    {
        simd_mask<_DATA_PER_STEP> __m = (__io_offset + __lane_id + __s) < __n;
        __keys.template select<_DATA_PER_STEP, 1>(__s) =
            merge(__utils::__gather<_KeyT, _DATA_PER_STEP>(__input, __lane_id, __io_offset + __s, __m),
                  simd<_KeyT, _DATA_PER_STEP>(__utils::__sort_identity<_KeyT, _IsAscending>()), __m);
    }

    for (uint32_t __stage = 0; __stage < _STAGES; __stage++)
    {
        __bins = __utils::__get_bucket<_MASK>(__utils::__order_preserving_cast<_IsAscending>(__keys), __stage * _RadixBits);

        __bin_offset = 0;
#pragma unroll
        for (uint32_t __s = 0; __s < _DataPerWorkItem; __s += 1)
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
            for (uint32_t __s = 0; __s < _BIN_COUNT; __s += 128)
            {
                __utils::_BlockStoreSlm<uint32_t, 64>(
                    __slm_bin_hist_this_thread + __s * sizeof(_hist_t),
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<uint32_t>());
            }
            barrier();
            constexpr uint32_t _BIN_SUMMARY_GROUP_SIZE = 8;
            if (__local_tid < _BIN_SUMMARY_GROUP_SIZE)
            {
                constexpr uint32_t _BIN_WIDTH = _BIN_COUNT / _BIN_SUMMARY_GROUP_SIZE;
                constexpr uint32_t _BIN_WIDTH_UD = _BIN_WIDTH * sizeof(_hist_t) / sizeof(uint32_t);
                uint32_t __slm_bin_hist_summary_offset = __local_tid * _BIN_WIDTH * sizeof(_hist_t);
                simd<_hist_t, _BIN_WIDTH> ___thread_grf_hist_summary;
                simd<uint32_t, _BIN_WIDTH_UD> __tmp;

                ___thread_grf_hist_summary.template bit_cast_view<uint32_t>() =
                    __utils::_BlockLoadSlm<uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                __slm_bin_hist_summary_offset += _HIST_STRIDE;
                for (uint32_t __s = 1; __s < _WorkGroupSize - 1; __s++)
                {
                    __tmp = __utils::_BlockLoadSlm<uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                    ___thread_grf_hist_summary += __tmp.template bit_cast_view<_hist_t>();
                    __utils::_BlockStoreSlm<uint32_t, _BIN_WIDTH_UD>(
                        __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<uint32_t>());
                    __slm_bin_hist_summary_offset += _HIST_STRIDE;
                }
                __tmp = __utils::_BlockLoadSlm<uint32_t, _BIN_WIDTH_UD>(__slm_bin_hist_summary_offset);
                ___thread_grf_hist_summary += __tmp.template bit_cast_view<_hist_t>();
                ___thread_grf_hist_summary = __utils::__scan<_hist_t, _hist_t>(___thread_grf_hist_summary);
                __utils::_BlockStoreSlm<uint32_t, _BIN_WIDTH_UD>(
                    __slm_bin_hist_summary_offset, ___thread_grf_hist_summary.template bit_cast_view<uint32_t>());
            }
            barrier();
            if (__local_tid == 0)
            {
                simd<_hist_t, _BIN_COUNT> __grf_hist_summary;
                simd<_hist_t, _BIN_COUNT + 1> __grf_hist_summary_scan;
#pragma unroll
                for (uint32_t __s = 0; __s < _BIN_COUNT; __s += 128)
                {
                    __grf_hist_summary.template select<128, 1>(__s).template bit_cast_view<uint32_t>() =
                        __utils::_BlockLoadSlm<uint32_t, 64>((_WorkGroupSize - 1) * _HIST_STRIDE + __s * sizeof(_hist_t));
                }
                __grf_hist_summary_scan[0] = 0;
                __grf_hist_summary_scan.template select<32, 1>(1) = __grf_hist_summary.template select<32, 1>(0);
#pragma unroll
                for (uint32_t __i = 32; __i < _BIN_COUNT; __i += 32)
                {
                    __grf_hist_summary_scan.template select<32, 1>(__i + 1) =
                        __grf_hist_summary.template select<32, 1>(__i) + __grf_hist_summary_scan[__i];
                }
#pragma unroll
                for (uint32_t __s = 0; __s < _BIN_COUNT; __s += 128)
                {
                    __utils::_BlockStoreSlm<uint32_t, 64>(
                        _BIN_HIST_SLM_SIZE + __s * sizeof(_hist_t),
                        __grf_hist_summary_scan.template select<128, 1>(__s).template bit_cast_view<uint32_t>());
                }
            }
            barrier();
            {
#pragma unroll
                for (uint32_t __s = 0; __s < _BIN_COUNT; __s += 128)
                {
                    __bin_offset.template select<128, 1>(__s).template bit_cast_view<uint32_t>() =
                        __utils::_BlockLoadSlm<uint32_t, 64>(_BIN_HIST_SLM_SIZE + __s * sizeof(_hist_t));
                }
                if (__local_tid > 0)
                {
#pragma unroll
                    for (uint32_t __s = 0; __s < _BIN_COUNT; __s += 128)
                    {
                        simd<_hist_t, 128> __group_local_sum;
                        __group_local_sum.template bit_cast_view<uint32_t>() =
                            __utils::_BlockLoadSlm<uint32_t, 64>((__local_tid - 1) * _HIST_STRIDE + __s * sizeof(_hist_t));
                        __bin_offset.template select<128, 1>(__s) += __group_local_sum;
                    }
                }
            }
            barrier();
        }

#pragma unroll
        for (uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
        {
            simd<uint16_t, _DATA_PER_STEP> __bins_uw = __bins.template select<_DATA_PER_STEP, 1>(__s);
            __write_addr.template select<_DATA_PER_STEP, 1>(__s) += __bin_offset.template iselect(__bins_uw);
        }

        if (__stage != _STAGES - 1)
        {
#pragma unroll
            for (uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
            {
                __utils::_VectorStore<_KeyT, 1, _DATA_PER_STEP>(
                    __write_addr.template select<_DATA_PER_STEP, 1>(__s) * sizeof(_KeyT),
                    __keys.template select<_DATA_PER_STEP, 1>(__s));
            }
            barrier();
            __keys = __utils::_BlockLoadSlm<_KeyT, _DataPerWorkItem>(__slm_reorder_this_thread);
        }
    }
#pragma unroll
    for (uint32_t __s = 0; __s < _DataPerWorkItem; __s += _DATA_PER_STEP)
    {
        __utils::__scatter<_KeyT, _DATA_PER_STEP>(__input, __write_addr.template select<_DATA_PER_STEP, 1>(__s),
                                             __keys.template select<_DATA_PER_STEP, 1>(__s),
                                             __write_addr.template select<_DATA_PER_STEP, 1>(__s) < __n);
    }
}

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------
template <typename... _Name>
class __esimd_radix_sort_one_wg;

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename _KernelName>
struct __radix_sort_one_wg_submitter;

template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename... _Name>
struct __radix_sort_one_wg_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Range>
    sycl::event
    operator()(sycl::queue __q, _Range&& __rng, ::std::size_t __n) const
    {
        sycl::nd_range<1> __nd_range{_WorkGroupSize, _WorkGroupSize};
        return __q.submit(
            [&](sycl::handler& __cgh)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __data = __rng.data();
                __cgh.parallel_for<_Name...>(
                    __nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                        __one_wg_kernel<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT>(__nd_item, __n,
                                                                                                         __data);
                    });
            });
    }
};

template <typename _KernelName, bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _Range>
sycl::event
__one_wg(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;
    using _EsimRadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__esimd_radix_sort_one_wg<_KernelName>>;

    return __radix_sort_one_wg_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT,
                                         _EsimRadixSortKernel>()(__q, ::std::forward<_Range>(__rng), __n);
}

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONE_WG_H
