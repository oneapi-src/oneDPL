// -*- C++ -*-
//===-- esimd_radix_sort_dispatchers.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_DISPATCHERS_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_DISPATCHERS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>
#include <cassert>

#include "../kernel_param.h"
#include "esimd_radix_sort_utils.h"
#include "esimd_radix_sort_submitters.h"
#include "esimd_radix_sort_kernels.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

namespace oneapi::dpl::experimental::kt::esimd::__impl
{
template <typename... _Name>
class __esimd_radix_sort_one_wg;

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

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeysRng>
sycl::event
__onesweep(sycl::queue __q, _KeysRng&& __keys_rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;

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

    constexpr ::std::uint32_t __hist_mem_size = __bin_count * sizeof(_GlobalHistT);
    constexpr ::std::uint32_t __global_hist_mem_size = __stage_count * __hist_mem_size;
    const ::std::uint32_t __group_hists_mem_size = __sweep_work_group_count * __stage_count * __hist_mem_size;
    const ::std::uint32_t __tmp_keys_mem_size =  __n * sizeof(_KeyT);
    const ::std::size_t __tmp_mem_size = __group_hists_mem_size + __global_hist_mem_size + __tmp_keys_mem_size;

    ::std::uint8_t* __p_tmp_mem = sycl::malloc_device<::std::uint8_t>(__tmp_mem_size, __q);
    // Memory to store global histograms, where each stage has its own histogram
    _GlobalHistT* __p_global_hist_all = reinterpret_cast<_GlobalHistT*>(__p_tmp_mem);
    // Memory to store group historgrams, which contain offsets relative to "previous" groups
    _GlobalHistT* __p_group_hists_all = reinterpret_cast<_GlobalHistT*>(__p_tmp_mem + __global_hist_mem_size);
    // Memory to store intermediate results of sorting
    _KeyT* __p_keys_tmp = reinterpret_cast<_KeyT*>(__p_tmp_mem + __global_hist_mem_size + __group_hists_mem_size);

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _KeyT*>();
    auto __keys_tmp_rng = __keep(__p_keys_tmp, __p_keys_tmp + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event __event_chain = __q.memset(__p_tmp_mem, 0, __global_hist_mem_size + __group_hists_mem_size);

    __event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size,
                                                  __is_ascending, _EsimdRadixSortHistogram>()(__q, __keys_rng, __p_global_hist_all,
                                                                                            __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _EsimdRadixSortScan>()(__q, __p_global_hist_all,
                                                                                                __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        _GlobalHistT* __p_global_hist = __p_global_hist_all + __bin_count * __stage;
        _GlobalHistT* __p_group_hists = __p_group_hists_all + __sweep_work_group_count * __bin_count * __stage;
        if ((__stage % 2) == 0)
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _EsimdRadixSortSweepEven>()(
                __q, __keys_rng, __keys_tmp_rng, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _EsimdRadixSortSweepOdd>()(
                __q, __keys_tmp_rng, __keys_rng, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
        }
    }

    if constexpr (__stage_count % 2 != 0)
    {
        __event_chain =
            __radix_sort_copyback_submitter<_KeyT, _EsimdRadixSortCopyback>()(__q, __keys_tmp_rng, __keys_rng, __n, __event_chain);
    }

    __event_chain = __q.submit(
        [__event_chain, __p_tmp_mem, __q](sycl::handler& __cgh)
        {
            __cgh.depends_on(__event_chain);
            __cgh.host_task([=]() { sycl::free(__p_tmp_mem, __q); });
        });

    return __event_chain;
}

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeysRng, typename _ValsRng>
sycl::event
__onesweep_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;
    using _ValT = oneapi::dpl::__internal::__value_t<_ValsRng>;

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

    using _GlobalHistT = ::std::uint32_t;
    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr ::std::uint32_t __hist_work_group_count = 64;
    constexpr ::std::uint32_t __hist_work_group_size = 64;

    const ::std::uint32_t __sweep_work_group_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __data_per_work_item);
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);

    constexpr ::std::uint32_t __hist_mem_size = __bin_count * sizeof(_GlobalHistT);
    constexpr ::std::uint32_t __global_hist_mem_size = __stage_count * __hist_mem_size;
    const ::std::uint32_t __group_hists_mem_size = __sweep_work_group_count * __stage_count * __hist_mem_size;
    const ::std::uint32_t __tmp_keys_mem_size =  __n * sizeof(_KeyT);
    const ::std::uint32_t __tmp_vals_mem_size =  __n * sizeof(_ValT);
    const ::std::size_t __tmp_mem_size = __group_hists_mem_size + __global_hist_mem_size + __tmp_keys_mem_size + __tmp_vals_mem_size;

    ::std::uint8_t* __p_tmp_mem = sycl::malloc_device<::std::uint8_t>(__tmp_mem_size, __q);
    // Memory to store global histograms, where each stage has its own histogram
    _GlobalHistT* __p_global_hist_all = reinterpret_cast<_GlobalHistT*>(__p_tmp_mem);
    // Memory to store group histograms, which contain offsets relative to "previous" groups
    _GlobalHistT* __p_group_hists_all = reinterpret_cast<_GlobalHistT*>(__p_tmp_mem + __global_hist_mem_size);
    // Memory to store intermediate results of sorting
    _KeyT* __p_keys_tmp = reinterpret_cast<_KeyT*>(__p_tmp_mem + __global_hist_mem_size + __group_hists_mem_size);
    _ValT* __p_vals_tmp = reinterpret_cast<_ValT*>(__p_tmp_mem + __global_hist_mem_size + __group_hists_mem_size + __tmp_keys_mem_size);

    auto __keys_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _KeyT*>();
    auto __keys_tmp_rng = __keys_tmp_keep(__p_keys_tmp, __p_keys_tmp + __n).all_view();

    auto __vals_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _ValT*>();
    auto __vals_tmp_rng = __vals_tmp_keep(__p_vals_tmp, __p_vals_tmp + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    sycl::event __event_chain = __q.memset(__p_tmp_mem, 0, __global_hist_mem_size + __group_hists_mem_size);

    __event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size,
                                                  __is_ascending, _EsimdRadixSortHistogram>()(__q, __keys_rng, __p_global_hist_all,
                                                                                            __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _EsimdRadixSortScan>()(__q, __p_global_hist_all,
                                                                                                __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        _GlobalHistT* __p_global_hist = __p_global_hist_all + __bin_count * __stage;
        _GlobalHistT* __p_group_hists = __p_group_hists_all + __sweep_work_group_count * __bin_count * __stage;
        if ((__stage % 2) == 0)
        {
            __event_chain = __radix_sort_onesweep_by_key_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _ValT, _EsimdRadixSortSweepEven>()(
                __q, __keys_rng, __keys_tmp_rng, __vals_rng, __vals_tmp_rng, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_by_key_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                          _KeyT, _ValT, _EsimdRadixSortSweepOdd>()(
                __q, __keys_tmp_rng, __keys_rng, __vals_tmp_rng, __vals_rng, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
        }
    }

    if constexpr (__stage_count % 2 != 0)
    {
        __event_chain =
            __radix_sort_by_key_copyback_submitter<_KeyT, _ValT, _EsimdRadixSortCopyback>()(__q, __keys_tmp_rng, __keys_rng, __vals_tmp_rng, __vals_rng, __n, __event_chain);
    }

    __event_chain = __q.submit(
        [__event_chain, __p_tmp_mem, __q](sycl::handler& __cgh)
        {
            __cgh.depends_on(__event_chain);
            __cgh.host_task([=]() { sycl::free(__p_tmp_mem, __q); });
        });

    return __event_chain;
}

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool __is_ascending, ::std::uint8_t __radix_bits, typename _KernelParam, typename _Range>
sycl::event
__radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param)
{
    static_assert(__radix_bits == 8);

    static_assert(32 <= __param.data_per_workitem && __param.data_per_workitem <= 512 &&
                  __param.data_per_workitem % 32 == 0);

    const ::std::size_t __n = __rng.size();
    assert(__n > 1);

    // _PRINT_INFO_IN_DEBUG_MODE(__exec); TODO: extend the utility to work with queues
    constexpr auto __data_per_workitem = _KernelParam::data_per_workitem;
    constexpr auto __workgroup_size = _KernelParam::workgroup_size;
    using _KernelName = typename _KernelParam::kernel_name;

    constexpr ::std::uint32_t __one_wg_cap = __data_per_workitem * __workgroup_size;
    if (__n <= __one_wg_cap)
    {
        // TODO: support different RadixBits values (only 7 or 8 are currently supported), WorkGroupSize
        return __one_wg<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
            __q, ::std::forward<_Range>(__rng), __n);
    }
    else
    {
        // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
        // TODO: support different RadixBits, WorkGroupSize
        return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
            __q, ::std::forward<_Range>(__rng), __n);
    }
}

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool __is_ascending, ::std::uint8_t __radix_bits, typename _KernelParam, typename _KeysRng, typename _ValsRng>
sycl::event
__radix_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, _KernelParam __param)
{
    static_assert(__radix_bits == 8);

    static_assert(32 <= __param.data_per_workitem && __param.data_per_workitem <= 512 &&
                  __param.data_per_workitem % 32 == 0);

    const ::std::size_t __n = __keys_rng.size();
    assert(__n > 1);

    // _PRINT_INFO_IN_DEBUG_MODE(__exec); TODO: extend the utility to work with queues
    constexpr auto __data_per_workitem = _KernelParam::data_per_workitem;
    constexpr auto __workgroup_size = _KernelParam::workgroup_size;
    using _KernelName = typename _KernelParam::kernel_name;

    // TODO: enable sort_by_key for one-work-group implementation
    return __onesweep_by_key<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
        __q, ::std::forward<_KeysRng>(__keys_rng), ::std::forward<_ValsRng>(__vals_rng), __n);
}

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_DISPATCHERS_H
