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
#include <type_traits> // std::alignment_of_v, std::is_void_v
#include <memory> // std::align

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
          ::std::uint16_t __work_group_size, typename _RngPack>
sycl::event
__one_wg(sycl::queue __q, _RngPack&& __pack, ::std::size_t __n)
{
    using _KeyT = typename _RngPack::_KeyT;
    using _EsimRadixSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__esimd_radix_sort_one_wg<_KernelName>>;

    return __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                         _EsimRadixSortKernel>()(__q, ::std::forward<_RngPack>(__pack), __n);
}

template<typename _HistT, typename _KeyT, typename _ValT = void>
class __onesweep_memory_holder
{
    static constexpr bool __has_values = !::std::is_void_v<_ValT>;

    ::std::uint8_t* __m_raw_mem_ptr = nullptr;
    // Memory to store global histograms, where each stage has its own histogram
    _HistT* __m_global_hist_ptr = nullptr;
    // Memory to store group historgrams, which contain offsets relative to "previous" groups
    _HistT* __m_group_hist_ptr = nullptr;
    // Memory to store intermediate results of sorting
    _KeyT* __m_keys_ptr = nullptr;
    _ValT* __m_vals_ptr = nullptr;

    ::std::size_t __m_raw_mem_bytes = 0;
    ::std::size_t __m_keys_bytes = 0;
    ::std::size_t __m_vals_bytes = 0;
    ::std::size_t __m_global_hist_bytes = 0;
    ::std::size_t __m_group_hist_bytes = 0;

    sycl::queue __m_q;

    void __calculate_raw_memory_amount() noexcept
    {
        // Extra bytes are added for potentiall padding
        __m_raw_mem_bytes = __m_keys_bytes + __m_global_hist_bytes + __m_group_hist_bytes + sizeof(_KeyT);
        if constexpr (__has_values)
        {
            __m_raw_mem_bytes += (__m_vals_bytes + sizeof(_ValT));
        }
    }

    void __allocate_raw_memory()
    {
        // Non-typed allocation is guaranteed to be aligned for any fundamental type according to SYCL spec
        // TODO: handle a case when malloc_device fails to allocate the memory
        void* __mem = sycl::malloc_device(__m_raw_mem_bytes, __m_q);
        __m_raw_mem_ptr = reinterpret_cast<::std::uint8_t*>(__mem);
    }

    void __appoint_aligned_memory_regions()
    {
        // It assumes that the raw pointer is already aligned for _HistT
        __m_global_hist_ptr = reinterpret_cast<_HistT*>(__m_raw_mem_ptr);
        __m_group_hist_ptr = reinterpret_cast<_HistT*>(__m_raw_mem_ptr + __m_global_hist_bytes);

        void* __base_ptr = reinterpret_cast<void*>(__m_raw_mem_ptr + __m_global_hist_bytes + __m_group_hist_bytes);
        ::std::size_t __remainder = __m_raw_mem_bytes - (__m_global_hist_bytes + __m_group_hist_bytes);
        void* __aligned_ptr = ::std::align(::std::alignment_of_v<_KeyT>, __m_keys_bytes, __base_ptr, __remainder);
        __m_keys_ptr = reinterpret_cast<_KeyT*>(__aligned_ptr);

        if constexpr (__has_values)
        {
            __base_ptr = reinterpret_cast<void*>(__m_keys_ptr + __m_keys_bytes / sizeof(_KeyT));
            __remainder = __m_raw_mem_bytes - (__m_global_hist_bytes + __m_group_hist_bytes + __m_keys_bytes);
            __aligned_ptr = ::std::align(::std::alignment_of_v<_ValT>, __m_vals_bytes, __base_ptr, __remainder);
            __m_vals_ptr = reinterpret_cast<_ValT*>(__aligned_ptr);
        }
    }

public:
    __onesweep_memory_holder(sycl::queue __q): __m_q(__q) {}

    void __keys_alloc_count(::std::size_t __key_count) noexcept
    {
        __m_keys_bytes = __key_count * sizeof(_KeyT);
    }
    void __vals_alloc_count(::std::size_t __values_count) noexcept
    {
        __m_vals_bytes = __values_count * sizeof(_ValT);
    }
    void __global_hist_item_alloc_count(::std::size_t __global_hist_item_count) noexcept
    {
        __m_global_hist_bytes = __global_hist_item_count * sizeof(_HistT);
    }
    void __group_hist_item_alloc_count(::std::size_t __group_hist_item_count) noexcept
    {
        __m_group_hist_bytes = __group_hist_item_count * sizeof(_HistT);
    }

    _KeyT* __keys_ptr() noexcept { return __m_keys_ptr; }
    _ValT* __vals_ptr() noexcept { return __m_vals_ptr; }
    _HistT* __global_hist_ptr() noexcept { return __m_global_hist_ptr; }
    _HistT* __group_hist_ptr() noexcept{ return __m_group_hist_ptr; }

    void __allocate()
    {
        __calculate_raw_memory_amount();
        __allocate_raw_memory();
        __appoint_aligned_memory_regions();
    }

    sycl::event __async_deallocate(sycl::event __dep_event)
    {
        auto __dealloc_task = [__q = __m_q, __e = __dep_event, __mem = __m_raw_mem_ptr](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            __cgh.host_task([=]() { sycl::free(__mem, __q); });
        };
        return __m_q.submit(__dealloc_task);
    }
};

template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _RngPack>
sycl::event
__onesweep(sycl::queue __q, _RngPack&& __pack, ::std::size_t __n)
{
    using _KeyT = typename _RngPack::_KeyT;
    using _ValT = typename _RngPack::_ValT;

    using _EsimdRadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __esimd_radix_sort_onesweep_histogram<_KernelName>>;
    using _EsimdRadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __esimd_radix_sort_onesweep_scan<_KernelName>>;
    using _EsimdRadixSortSweepEven = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        ::std::conditional_t<_RngPack::__has_values, __esimd_radix_sort_onesweep_even_by_key<_KernelName>,
                                                     __esimd_radix_sort_onesweep_even<_KernelName>>>;
    using _EsimdRadixSortSweepOdd = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        ::std::conditional_t<_RngPack::__has_values, __esimd_radix_sort_onesweep_odd_by_key<_KernelName>,
                                                     __esimd_radix_sort_onesweep_odd<_KernelName>>>;
    using _EsimdRadixSortCopyback = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        ::std::conditional_t<_RngPack::__has_values, __esimd_radix_sort_onesweep_copyback_by_key<_KernelName>,
                                                      __esimd_radix_sort_onesweep_copyback<_KernelName>>>;

    using _GlobalHistT = ::std::uint32_t;
    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    const ::std::uint32_t __sweep_work_group_count = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __data_per_work_item);
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);

    constexpr ::std::uint32_t __global_hist_item_count = __bin_count * __stage_count;
    const ::std::uint32_t __group_hist_item_count = __bin_count * __stage_count * __sweep_work_group_count;

    __onesweep_memory_holder<_GlobalHistT, _KeyT, _ValT> __mem_holder(__q);
    __mem_holder.__keys_alloc_count(__n);
    if constexpr (_RngPack::__has_values)
    {
        __mem_holder.__vals_alloc_count(__n);
    }
    __mem_holder.__global_hist_item_alloc_count(__global_hist_item_count);
    __mem_holder.__group_hist_item_alloc_count(__group_hist_item_count);
    __mem_holder.__allocate();

    auto __get_tmp_pack = [&]() {
        auto __keys_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                        _KeyT*>();
        auto __keys_tmp_rng = __keys_tmp_keep(__mem_holder.__keys_ptr(), __mem_holder.__keys_ptr() + __n).all_view();
        if constexpr (_RngPack::__has_values)
        {
            auto __vals_tmp_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                                _ValT*>();
            auto __vals_tmp_rng = __vals_tmp_keep(__mem_holder.__vals_ptr(), __mem_holder.__vals_ptr() + __n).all_view();
             return __utils::__rng_pack(::std::move(__keys_tmp_rng), ::std::move(__vals_tmp_rng));
        }
        else
        {
            return __utils::__rng_pack(::std::move(__keys_tmp_rng));
        }
    };
    auto __tmp_pack = __get_tmp_pack();

    // TODO: check if it is more performant to fill it inside the histgogram kernel
    // This line assumes that global and group histograms are stored sequentially
    sycl::event __event_chain = __q.memset(__mem_holder.__global_hist_ptr(), 0,
                                          (__global_hist_item_count + __group_hist_item_count) * sizeof(_GlobalHistT));

    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    constexpr ::std::uint32_t __hist_work_group_count = 64;
    constexpr ::std::uint32_t __hist_work_group_size = 64;
    __event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, __radix_bits, __stage_count, __hist_work_group_count, __hist_work_group_size,
                                                  __is_ascending, _EsimdRadixSortHistogram>()(__q, __pack.__keys_rng(), __mem_holder.__global_hist_ptr(),
                                                                                            __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _EsimdRadixSortScan>()(__q, __mem_holder.__global_hist_ptr(),
                                                                                                __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        _GlobalHistT* __p_global_hist = __mem_holder.__global_hist_ptr() + __bin_count * __stage;
        _GlobalHistT* __p_group_hists = __mem_holder.__group_hist_ptr() + __sweep_work_group_count * __bin_count * __stage;
        if ((__stage % 2) == 0)
        {
            if constexpr (_RngPack::__has_values)
            {
                __event_chain = __radix_sort_onesweep_by_key_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                            _KeyT, _ValT, _EsimdRadixSortSweepEven>()(
                    __q, __pack, __tmp_pack, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
            }
            else
            {
                __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                            _KeyT, _EsimdRadixSortSweepEven>()(
                    __q, __pack, __tmp_pack, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
            }
        }
        else
        {
            if constexpr (_RngPack::__has_values)
            {
                __event_chain = __radix_sort_onesweep_by_key_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                            _KeyT, _ValT, _EsimdRadixSortSweepOdd>()(
                    __q, __tmp_pack, __pack, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
            }
            else
            {
                __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                                            _KeyT, _EsimdRadixSortSweepOdd>()(
                    __q, __tmp_pack, __pack, __p_global_hist, __p_group_hists, __sweep_work_group_count, __n, __stage, __event_chain);
            }
        }
    }

    if constexpr (__stage_count % 2 != 0)
    {
        if constexpr (_RngPack::__has_values)
        {
            __event_chain =
                __radix_sort_by_key_copyback_submitter<_KeyT, _ValT, _EsimdRadixSortCopyback>()(__q, __tmp_pack, __pack, __n, __event_chain);
        }
        else
        {
            __event_chain =
                __radix_sort_copyback_submitter<_KeyT, _EsimdRadixSortCopyback>()(__q, __tmp_pack, __pack, __n, __event_chain);
        }
    }

    return __mem_holder.__async_deallocate(__event_chain);
}

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool __is_ascending, ::std::uint8_t __radix_bits, typename _RngPack, typename _KernelParam>
sycl::event
__radix_sort(sycl::queue __q, _RngPack&& __pack, _KernelParam __param)
{
    static_assert(__radix_bits == 8);

    static_assert(32 <= __param.data_per_workitem && __param.data_per_workitem <= 512 &&
                  __param.data_per_workitem % 32 == 0);

    const ::std::size_t __n = __pack.__keys_rng().size();
    assert(__n > 1);

    // _PRINT_INFO_IN_DEBUG_MODE(__exec); TODO: extend the utility to work with queues
    constexpr auto __data_per_workitem = _KernelParam::data_per_workitem;
    constexpr auto __workgroup_size = _KernelParam::workgroup_size;
    using _KernelName = typename _KernelParam::kernel_name;

    // TODO: enable sort_by_key for one-work-group implementation
    if constexpr (_RngPack::__has_values)
    {
        return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
            __q, ::std::forward<_RngPack>(__pack), __n);
    }
    else
    {
        constexpr ::std::uint32_t __one_wg_cap = __data_per_workitem * __workgroup_size;
        if (__n <= __one_wg_cap)
        {
            // TODO: support different RadixBits values (only 7 or 8 are currently supported), WorkGroupSize
            return __one_wg<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
                __q, ::std::forward<_RngPack>(__pack), __n);
        }
        else
        {
            // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
            // TODO: support different RadixBits, WorkGroupSize
            return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
                __q, ::std::forward<_RngPack>(__pack), __n);
        }
    }
}

} // namespace experimental::kt::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_DISPATCHERS_H
