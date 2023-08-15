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

namespace oneapi::dpl::experimental::kt::esimd::impl
{
template <bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _KeyT, typename InputT>
void
one_wg_kernel(sycl::nd_item<1> idx, uint32_t n, const InputT& input)
{
    using namespace sycl;
    using namespace __dpl_esimd_ns;
    using namespace __dpl_esimd_ens;

    using bin_t = uint16_t;
    using hist_t = uint16_t;
    using device_addr_t = uint32_t;

    constexpr uint32_t BIN_COUNT = 1 << _RadixBits;
    constexpr uint32_t NBITS = sizeof(_KeyT) * 8;
    constexpr uint32_t STAGES = oneapi::dpl::__internal::__dpl_ceiling_div(NBITS, _RadixBits);
    constexpr bin_t MASK = BIN_COUNT - 1;
    constexpr uint32_t HIST_STRIDE = sizeof(hist_t) * BIN_COUNT;

    constexpr uint32_t REORDER_SLM_SIZE = _DataPerWorkItem * sizeof(_KeyT) * _WorkGroupSize;
    constexpr uint32_t BIN_HIST_SLM_SIZE = HIST_STRIDE * _WorkGroupSize;
    constexpr uint32_t INCOMING_OFFSET_SLM_SIZE = (BIN_COUNT + 1) * sizeof(hist_t);

    // max SLM is 256 * 4 * 64 + 256 * 2 * 64 + 257*2, 97KB, when  _DataPerWorkItem = 256, BIN_COUNT = 256
    // to support 512 processing size, we can use all SLM as reorder buffer with cost of more barrier
    slm_init(std::max(REORDER_SLM_SIZE, BIN_HIST_SLM_SIZE + INCOMING_OFFSET_SLM_SIZE));

    uint32_t local_tid = idx.get_local_linear_id();
    uint32_t slm_reorder_start = 0;
    uint32_t slm_bin_hist_start = 0;
    uint32_t slm_incoming_offset = slm_bin_hist_start + BIN_HIST_SLM_SIZE;

    uint32_t slm_reorder_this_thread = slm_reorder_start + local_tid * _DataPerWorkItem * sizeof(_KeyT);
    uint32_t slm_bin_hist_this_thread = slm_bin_hist_start + local_tid * HIST_STRIDE;

    simd<hist_t, BIN_COUNT> bin_offset;
    simd<device_addr_t, _DataPerWorkItem> write_addr;
    simd<_KeyT, _DataPerWorkItem> keys;
    simd<bin_t, _DataPerWorkItem> bins;
    simd<device_addr_t, DATA_PER_STEP> lane_id(0, 1);

    device_addr_t io_offset = _DataPerWorkItem * local_tid;

#pragma unroll
    for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
    {
        simd_mask<DATA_PER_STEP> m = (io_offset + lane_id + s) < n;
        keys.template select<DATA_PER_STEP, 1>(s) =
            merge(utils::gather<_KeyT, DATA_PER_STEP>(input, lane_id, io_offset + s, m),
                  simd<_KeyT, DATA_PER_STEP>(utils::__sort_identity<_KeyT, _IsAscending>()), m);
    }

    for (uint32_t stage = 0; stage < STAGES; stage++)
    {
        bins = utils::__get_bucket<MASK>(utils::__order_preserving_cast<_IsAscending>(keys), stage * _RadixBits);

        bin_offset = 0;
#pragma unroll
        for (uint32_t s = 0; s < _DataPerWorkItem; s += 1)
        {
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
#pragma unroll
            for (uint32_t s = 0; s < BIN_COUNT; s += 128)
            {
                utils::BlockStore<uint32_t, 64>(
                    slm_bin_hist_this_thread + s * sizeof(hist_t),
                    bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>());
            }
            barrier();
            constexpr uint32_t BIN_SUMMARY_GROUP_SIZE = 8;
            if (local_tid < BIN_SUMMARY_GROUP_SIZE)
            {
                constexpr uint32_t BIN_WIDTH = BIN_COUNT / BIN_SUMMARY_GROUP_SIZE;
                constexpr uint32_t BIN_WIDTH_UD = BIN_WIDTH * sizeof(hist_t) / sizeof(uint32_t);
                uint32_t slm_bin_hist_summary_offset = slm_bin_hist_start + local_tid * BIN_WIDTH * sizeof(hist_t);
                simd<hist_t, BIN_WIDTH> thread_grf_hist_summary;
                simd<uint32_t, BIN_WIDTH_UD> tmp;

                thread_grf_hist_summary.template bit_cast_view<uint32_t>() =
                    utils::BlockLoad<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                slm_bin_hist_summary_offset += HIST_STRIDE;
                for (uint32_t s = 1; s < _WorkGroupSize - 1; s++)
                {
                    tmp = utils::BlockLoad<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                    thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                    utils::BlockStore<uint32_t, BIN_WIDTH_UD>(
                        slm_bin_hist_summary_offset, thread_grf_hist_summary.template bit_cast_view<uint32_t>());
                    slm_bin_hist_summary_offset += HIST_STRIDE;
                }
                tmp = utils::BlockLoad<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset);
                thread_grf_hist_summary += tmp.template bit_cast_view<hist_t>();
                thread_grf_hist_summary = utils::scan<hist_t, hist_t>(thread_grf_hist_summary);
                utils::BlockStore<uint32_t, BIN_WIDTH_UD>(slm_bin_hist_summary_offset,
                                                          thread_grf_hist_summary.template bit_cast_view<uint32_t>());
            }
            barrier();
            if (local_tid == 0)
            {
                simd<hist_t, BIN_COUNT> grf_hist_summary;
                simd<hist_t, BIN_COUNT + 1> grf_hist_summary_scan;
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    grf_hist_summary.template select<128, 1>(s).template bit_cast_view<uint32_t>() =
                        utils::BlockLoad<uint32_t, 64>(slm_bin_hist_start + (_WorkGroupSize - 1) * HIST_STRIDE +
                                                       s * sizeof(hist_t));
                }
                grf_hist_summary_scan[0] = 0;
                grf_hist_summary_scan.template select<32, 1>(1) = grf_hist_summary.template select<32, 1>(0);
#pragma unroll
                for (uint32_t i = 32; i < BIN_COUNT; i += 32)
                {
                    grf_hist_summary_scan.template select<32, 1>(i + 1) =
                        grf_hist_summary.template select<32, 1>(i) + grf_hist_summary_scan[i];
                }
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    utils::BlockStore<uint32_t, 64>(
                        slm_incoming_offset + s * sizeof(hist_t),
                        grf_hist_summary_scan.template select<128, 1>(s).template bit_cast_view<uint32_t>());
                }
            }
            barrier();
            {
#pragma unroll
                for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                {
                    bin_offset.template select<128, 1>(s).template bit_cast_view<uint32_t>() =
                        utils::BlockLoad<uint32_t, 64>(slm_incoming_offset + s * sizeof(hist_t));
                }
                if (local_tid > 0)
                {
#pragma unroll
                    for (uint32_t s = 0; s < BIN_COUNT; s += 128)
                    {
                        simd<hist_t, 128> group_local_sum;
                        group_local_sum.template bit_cast_view<uint32_t>() = utils::BlockLoad<uint32_t, 64>(
                            slm_bin_hist_start + (local_tid - 1) * HIST_STRIDE + s * sizeof(hist_t));
                        bin_offset.template select<128, 1>(s) += group_local_sum;
                    }
                }
            }
            barrier();
        }

#pragma unroll
        for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
        {
            simd<uint16_t, DATA_PER_STEP> bins_uw = bins.template select<DATA_PER_STEP, 1>(s);
            write_addr.template select<DATA_PER_STEP, 1>(s) += bin_offset.template iselect(bins_uw);
        }

        if (stage != STAGES - 1)
        {
#pragma unroll
            for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
            {
                utils::VectorStore<_KeyT, 1, DATA_PER_STEP>(
                    write_addr.template select<DATA_PER_STEP, 1>(s) * sizeof(_KeyT) + slm_reorder_start,
                    keys.template select<DATA_PER_STEP, 1>(s));
            }
            barrier();
            keys = utils::BlockLoad<_KeyT, _DataPerWorkItem>(slm_reorder_this_thread);
        }
    }
#pragma unroll
    for (uint32_t s = 0; s < _DataPerWorkItem; s += DATA_PER_STEP)
    {
        utils::scatter<_KeyT, DATA_PER_STEP>(input, write_addr.template select<DATA_PER_STEP, 1>(s),
                                             keys.template select<DATA_PER_STEP, 1>(s),
                                             write_addr.template select<DATA_PER_STEP, 1>(s) < n);
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
                        one_wg_kernel<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT>(__nd_item, __n,
                                                                                                         __data);
                    });
            });
    }
};

template <typename _KernelName, bool _IsAscending, ::std::uint8_t _RadixBits, ::std::uint16_t _DataPerWorkItem,
          ::std::uint16_t _WorkGroupSize, typename _Range>
sycl::event
one_wg(sycl::queue __q, _Range&& __rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;
    using _EsimRadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__esimd_radix_sort_one_wg<_KernelName>>;

    return __radix_sort_one_wg_submitter<_IsAscending, _RadixBits, _DataPerWorkItem, _WorkGroupSize, _KeyT,
                                         _EsimRadixSortKernel>()(__q, ::std::forward<_Range>(__rng), __n);
}

} // namespace oneapi::dpl::experimental::kt::esimd::impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONE_WG_H
