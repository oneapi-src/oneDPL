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
#include "esimd_radix_sort_onesweep_kernels.h"

#include <cstdint>

namespace oneapi::dpl::experimental::esimd::impl
{

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

template <typename... _Name>
class __esimd_radix_sort_onesweep_even_by_key;

template <typename... _Name>
class __esimd_radix_sort_onesweep_odd_by_key;

template <typename... _Name>
class __esimd_radix_sort_onesweep_copyback_by_key;

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

template <typename KeyT, typename ValueT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          bool IsAscending, typename _KernelName>
struct __radix_sort_onesweep_submitter_by_key;

template <typename KeyT, typename ValueT, ::std::uint32_t RADIX_BITS, ::std::uint32_t THREAD_PER_TG, ::std::uint32_t PROCESS_SIZE,
          bool IsAscending, typename... _Name>
struct __radix_sort_onesweep_submitter_by_key<KeyT, ValueT, RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE, IsAscending,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KeysInT, typename _KeysOutT, typename _ValuesInT, typename _ValuesOutT, typename _TmpData>
    sycl::event
    operator()(sycl::queue& __q, _KeysInT& __keys_in, _KeysOutT& __keys_out, _ValuesInT& __values_in, _ValuesOutT& __values_out, const _TmpData& __tmp_data,
               ::std::uint32_t __sweep_tg_count, ::std::size_t __n, ::std::uint32_t __stage, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_tg_count * THREAD_PER_TG, THREAD_PER_TG);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_in, __keys_out, __values_in, __values_out);
            auto __keys_in_data = __keys_in.data();
            auto __keys_out_data = __keys_out.data();
            auto __values_in_data = __values_in.data();
            auto __values_out_data = __values_out.data();
            __cgh.depends_on(__e);
            radix_sort_onesweep_slm_reorder_kernel_by_key<KeyT, ValueT,
                                                   decltype(__keys_in_data), decltype(__keys_out_data),
                                                   decltype(__values_in_data), decltype(__values_out_data),
                                                   RADIX_BITS, THREAD_PER_TG, PROCESS_SIZE, IsAscending>
                K(__n, __stage, __keys_in_data, __keys_out_data, __values_in_data, __values_out_data, __tmp_data);
            __cgh.parallel_for<_Name...>(__nd_range, K);
        });
    }
};

template <typename KeyT, typename _KernelName>
struct __radix_sort_copyback_submitter_by_key;

template <typename KeyT, typename... _Name>
struct __radix_sort_copyback_submitter_by_key<KeyT, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KeysRangeTmp, typename _KeysRange, typename _ValuesRangeTmp, typename _ValuesRange>
    sycl::event
    operator()(sycl::queue& __q, _KeysRangeTmp& __keys_tmp, _KeysRange& __keys, _ValuesRangeTmp& __values_tmp,
               _ValuesRange& __values, ::std::uint32_t __n, const sycl::event& __e) const
    {
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_tmp, __keys, __values_tmp, __values);
            // TODO: make sure that access is read_only for __tmp_data  and is write_only for __out_rng
            auto __keys_tmp_data = __keys_tmp.data();
            auto __keys_data = __keys.data();
            auto __values_tmp_data = __values_tmp.data();
            auto __values_data = __values.data();
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(sycl::range<1>{__n}, [=](sycl::item<1> __item) {
                auto __global_id = __item.get_linear_id();
                __keys_data[__global_id] = __keys_tmp_data[__global_id];
                __values_data[__global_id] = __values_tmp_data[__global_id];
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

template <typename _KernelName, typename KeyT, typename ValueT, typename KeysRange, typename ValuesRange, ::std::uint32_t RADIX_BITS,
          bool IsAscending, ::std::uint32_t PROCESS_SIZE>
void
onesweep_by_key(sycl::queue __q, KeysRange&& __keys, ValuesRange&& __values, ::std::size_t __n)
{
    using namespace sycl;
    using namespace __ESIMD_NS;

    using _EsimdRadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_histogram<_KernelName>>;
    using _EsimdRadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_scan<_KernelName>>;
    using _EsimdRadixSortSweepEven = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_even_by_key<_KernelName>>;
    using _EsimdRadixSortSweepOdd = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_odd_by_key<_KernelName>>;
    using _EsimdRadixSortCopyback = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __esimd_radix_sort_onesweep_copyback_by_key<_KernelName>>;

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

    const size_t hist_buffer_size = GLOBAL_OFFSET_SIZE + SYNC_BUFFER_SIZE;
    const size_t keys_buffer_size = __n * sizeof(KeyT);
    const size_t values_buffer_size = __n * sizeof(ValueT);
    const size_t tmp_buffer_size = hist_buffer_size + keys_buffer_size + values_buffer_size;

    uint8_t* p_tmp_memory = sycl::malloc_device<uint8_t>(tmp_buffer_size, __q);

    auto p_global_hist_memory = p_tmp_memory;
    auto p_global_hist = reinterpret_cast<global_hist_t*>(p_global_hist_memory);

    // Memory for storing values sorted for an iteration
    auto p_keys_tmp = reinterpret_cast<KeyT*>(p_tmp_memory + hist_buffer_size);
    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, decltype(p_keys_tmp)>();
    auto __keys_tmp = __keys_keep(p_keys_tmp, p_keys_tmp + __n).all_view();
    auto p_values_tmp = reinterpret_cast<ValueT*>(p_tmp_memory + hist_buffer_size + keys_buffer_size);
    auto __values_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, decltype(p_values_tmp)>();
    auto __values_tmp = __values_keep(p_values_tmp, p_values_tmp + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event event_chain = __q.memset(p_global_hist_memory, 0, hist_buffer_size);

    event_chain = __radix_sort_onesweep_histogram_submitter<
        KeyT, RADIX_BITS, STAGES, HW_TG_COUNT, THREAD_PER_TG, IsAscending, _EsimdRadixSortHistogram>()(
            __q, __keys, p_global_hist, __n, event_chain);

    event_chain = __radix_sort_onesweep_scan_submitter<STAGES, BINCOUNT, _EsimdRadixSortScan>()(
        __q, p_global_hist, __n, event_chain);

    for (uint32_t stage = 0; stage < STAGES; stage++) {
        if((stage % 2) == 0)
        {
            event_chain = __radix_sort_onesweep_submitter_by_key<
                    KeyT, ValueT, RADIX_BITS, THREAD_PER_TG, SWEEP_PROCESSING_SIZE, IsAscending, _EsimdRadixSortSweepEven>()(
                        __q, __keys, __keys_tmp, __values, __values_tmp, p_global_hist_memory, sweep_tg_count, __n, stage, event_chain);
        }
        else
        {
            event_chain = __radix_sort_onesweep_submitter_by_key<
                    KeyT, ValueT, RADIX_BITS, THREAD_PER_TG, SWEEP_PROCESSING_SIZE, IsAscending, _EsimdRadixSortSweepOdd>()(
                        __q, __keys_tmp, __keys, __values_tmp, __values, p_global_hist_memory, sweep_tg_count, __n, stage, event_chain);
        }
    }

    if constexpr (STAGES % 2 != 0)
    {
        event_chain = __radix_sort_copyback_submitter_by_key<KeyT, _EsimdRadixSortCopyback>()(
            __q, __keys_tmp, __keys, __values_tmp, __values, __n, event_chain);
    }
    event_chain.wait();

    sycl::free(p_tmp_memory, __q);
}

} // oneapi::dpl::experimental::esimd::impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_ONESWEEP_H
