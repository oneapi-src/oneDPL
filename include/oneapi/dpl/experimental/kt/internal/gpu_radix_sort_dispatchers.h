// -*- C++ -*-
//===-- gpu_radix_sort_dispatchers.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_GPU_RADIX_SORT_DISPATCHERS_H
#define _ONEDPL_KT_GPU_RADIX_SORT_DISPATCHERS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>
#include <cassert>

#include "../kernel_param.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "gpu_radix_sort_submitters.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{
template <typename... _Name>
class __gpu_radix_sort_onesweep_histogram;

template <typename... _Name>
class __gpu_radix_sort_onesweep_scan;

template <typename... _Name>
class __gpu_radix_sort_onesweep_even;

template <typename... _Name>
class __gpu_radix_sort_onesweep_odd;

template <typename... _Name>
class __gpu_radix_sort_onesweep_copyback;

template<typename _HistT, typename _KeyT, typename _WgCounterT = std::uint32_t, typename _ValT = void>
class __onesweep_memory_holder
{
    static constexpr bool __has_values = !::std::is_void_v<_ValT>;

    ::std::uint8_t* __m_raw_mem_ptr = nullptr;
    // Memory to store global histograms, where each stage has its own histogram
    _HistT* __m_global_hist_ptr = nullptr;
    // Memory to store group historgrams, which contain offsets relative to "previous" groups
    _HistT* __m_group_hist_ptr = nullptr;
    // Memory to store dynamic wg ID
    _WgCounterT* __m_dynamic_id_ptr = nullptr;
    // Memory to store intermediate results of sorting
    _KeyT* __m_keys_ptr = nullptr;
    _ValT* __m_vals_ptr = nullptr;

    ::std::size_t __m_raw_mem_bytes = 0;
    ::std::size_t __m_keys_bytes = 0;
    ::std::size_t __m_vals_bytes = 0;
    ::std::size_t __m_global_hist_bytes = 0;
    ::std::size_t __m_group_hist_bytes = 0;
    ::std::size_t __m_dynamic_id_bytes = 0;

    sycl::queue __m_q;

    void __calculate_raw_memory_amount() noexcept
    {
        // Extra bytes are added for potentiall padding
        __m_raw_mem_bytes = __m_keys_bytes + __m_global_hist_bytes + __m_group_hist_bytes + sizeof(_KeyT) +
                            __m_dynamic_id_bytes + sizeof(_WgCounterT);
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
        // It assumes that the raw pointer is already alligned for _HistT
        __m_global_hist_ptr = reinterpret_cast<_HistT*>(__m_raw_mem_ptr);
        __m_group_hist_ptr = reinterpret_cast<_HistT*>(__m_raw_mem_ptr + __m_global_hist_bytes);

        void* __base_ptr = reinterpret_cast<void*>(__m_raw_mem_ptr + __m_global_hist_bytes + __m_group_hist_bytes);
        ::std::size_t __remainder = __m_raw_mem_bytes - (__m_global_hist_bytes + __m_group_hist_bytes);
        void* __aligned_ptr = ::std::align(::std::alignment_of_v<_WgCounterT>, __m_dynamic_id_bytes, __base_ptr, __remainder);
        __m_dynamic_id_ptr = reinterpret_cast<_WgCounterT*>(__aligned_ptr);

        __base_ptr = reinterpret_cast<void*>(__m_dynamic_id_ptr + __m_dynamic_id_bytes / sizeof(_WgCounterT));
        __remainder = __m_raw_mem_bytes - (__m_global_hist_bytes + __m_group_hist_bytes + __m_dynamic_id_bytes); // TODO this is wrong - padding bytes!
        __aligned_ptr = ::std::align(::std::alignment_of_v<_KeyT>, __m_keys_bytes, __base_ptr, __remainder);
        __m_keys_ptr = reinterpret_cast<_KeyT*>(__aligned_ptr);

        if constexpr (__has_values)
        {
            __base_ptr = reinterpret_cast<void*>(__m_keys_ptr + __m_keys_bytes / sizeof(_KeyT));
            __remainder = __m_raw_mem_bytes - (__m_global_hist_bytes + __m_group_hist_bytes + __m_dynamic_id_bytes + __m_keys_bytes);
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
    void __dynamic_id_alloc_count(::std::size_t __dynamic_id_count) noexcept
    {
        __m_dynamic_id_bytes = __dynamic_id_count * sizeof(_WgCounterT);
    }

    _KeyT* __keys_ptr() noexcept { return __m_keys_ptr; }
    _ValT* __vals_ptr() noexcept { return __m_vals_ptr; }
    _HistT* __global_hist_ptr() noexcept { return __m_global_hist_ptr; }
    _HistT* __group_hist_ptr() noexcept{ return __m_group_hist_ptr; }
    _WgCounterT* __dynamic_id_ptr() noexcept { return __m_dynamic_id_ptr;}

    void __allocate()
    {
        __calculate_raw_memory_amount();
        __allocate_raw_memory();
        __appoint_aligned_memory_regions();
    }

    void
    __deallocate(sycl::event __dep_event)
    {
        __dep_event.wait();
        sycl::free(__m_raw_mem_ptr, __m_q);
    }

    sycl::event
    __async_deallocate(sycl::event __dep_event)
    {
        auto __dealloc_task = [__q = __m_q, __event = __dep_event, __mem = __m_raw_mem_ptr](sycl::handler& __cgh) {
            __cgh.depends_on(__event);
            __cgh.host_task([=]() { sycl::free(__mem, __q); });
        };
        return __m_q.submit(__dealloc_task);
    }
};

//Note: The function _onesweep maps to the `RadixSortKernel_Impl::process` method from onesweep
// For RadixSortKernel_Impl, the ctor takes the data & the size, and `process` takes the queue.
template <typename _KernelName, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeysRng>
sycl::event
__onesweep(sycl::queue __q, _KeysRng&& __keys_rng, ::std::size_t __n)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;

    using _GpuRadixSortHistogram = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __gpu_radix_sort_onesweep_histogram<_KernelName>>;
    using _GpuRadixSortScan = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __gpu_radix_sort_onesweep_scan<_KernelName>>;
    using _GpuRadixSortSweepEven = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __gpu_radix_sort_onesweep_even<_KernelName>>;
    using _GpuRadixSortSweepOdd = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __gpu_radix_sort_onesweep_odd<_KernelName>>;
    using _GpuRadixSortCopyback = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        __gpu_radix_sort_onesweep_copyback<_KernelName>>;

    using _GlobalHistT = ::std::uint32_t;
    using _WgCounterT = ::std::uint32_t;
    constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    const ::std::uint32_t __sweep_work_group_count =
        oneapi::dpl::__internal::__dpl_ceiling_div(__n, __work_group_size * __data_per_work_item);
    constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr ::std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);

    constexpr ::std::uint32_t __global_hist_item_count = __bin_count * __stage_count;
    const ::std::uint32_t __group_hist_item_count = __bin_count * __stage_count * __sweep_work_group_count;

    __onesweep_memory_holder<_GlobalHistT, _KeyT> __mem_holder(__q);
    __mem_holder.__keys_alloc_count(__n);
    __mem_holder.__global_hist_item_alloc_count(__global_hist_item_count);
    __mem_holder.__group_hist_item_alloc_count(__group_hist_item_count);
    __mem_holder.__dynamic_id_alloc_count(__stage_count);
    __mem_holder.__allocate();

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _KeyT*>();
    auto __keys_tmp_rng = __keep(__mem_holder.__keys_ptr(), __mem_holder.__keys_ptr() + __n).all_view();

    // TODO: check if it is more performant to fill it inside the histogram kernel
    // This line assumes that global and group histograms are stored sequentially
    sycl::event __event_chain = __q.memset(__mem_holder.__global_hist_ptr(), 0,
                                           (__global_hist_item_count + __group_hist_item_count) * sizeof(_GlobalHistT) +
                                               __stage_count * sizeof(std::uint32_t));
    // TODO: consider adding a more versatile API, e.g. passing special kernel_config parameters for histogram computation
    __event_chain =
        __radix_sort_onesweep_histogram_submitter<_KeyT, __radix_bits, __stage_count, __data_per_work_item,
                                                  __work_group_size, __is_ascending, _GpuRadixSortHistogram>()(
            __q, __keys_rng, __mem_holder.__global_hist_ptr(), __n, __event_chain);

    __event_chain = __radix_sort_onesweep_scan_submitter<__stage_count, __bin_count, _GpuRadixSortScan>()(
        __q, __mem_holder.__global_hist_ptr(), __n, __event_chain);

    for (::std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
    {
        _GlobalHistT* __p_global_hist = __mem_holder.__global_hist_ptr() + __bin_count * __stage;
        _GlobalHistT* __p_group_hists =
            __mem_holder.__group_hist_ptr() + __sweep_work_group_count * __bin_count * __stage;
        _WgCounterT* __p_dynamic_id = __mem_holder.__dynamic_id_ptr();
        if ((__stage % 2) == 0)
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item,
                                                            __work_group_size, _KeyT, _GpuRadixSortSweepEven>()(
                __q, __keys_rng, __keys_tmp_rng, __p_global_hist, __p_group_hists, __p_dynamic_id, __sweep_work_group_count, __n,
                __stage, __event_chain);
        }
        else
        {
            __event_chain = __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item,
                                                            __work_group_size, _KeyT, _GpuRadixSortSweepOdd>()(
                __q, __keys_tmp_rng, __keys_rng, __p_global_hist, __p_group_hists, __p_dynamic_id, __sweep_work_group_count, __n,
                __stage, __event_chain);
        }
    }

    if constexpr (__stage_count % 2 != 0)
    {
        __event_chain = __radix_sort_copyback_submitter<_KeyT, _GpuRadixSortCopyback>()(__q, __keys_tmp_rng, __keys_rng,
                                                                                        __n, __event_chain);
    }

    if (0)
    {
        return __mem_holder.__async_deallocate(__event_chain);
    }
    else
    {
        //Cost of host thread creation can outweight perf benefits here
        __mem_holder.__deallocate(__event_chain);
        return __event_chain;
    }
}

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool __is_ascending, ::std::uint8_t __radix_bits, typename _KernelParam, typename _Range>
sycl::event
__radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param)
{
    const ::std::size_t __n = __rng.size();
    assert(__n > 1);

    constexpr auto __data_per_workitem = _KernelParam::data_per_workitem;
    constexpr auto __workgroup_size = _KernelParam::workgroup_size;
    using _KernelName = typename _KernelParam::kernel_name;

    static_assert(__workgroup_size % SUBGROUP_SIZE == 0 && "This kernel requires a multiple of 32 work-group size.");
    static_assert(__workgroup_size <= 256 && "This kernel requires too many registers per thread for the specified work-group size.");
    static_assert(__data_per_workitem % 4 == 0 && "__data_per_workitem must be a multiple of 4 for load vectorization");
    static_assert(__workgroup_size * __data_per_workitem <= std::numeric_limits<std::uint16_t>::max());

    // Check local memory size against device max
    constexpr auto __req_slm_size =
        sizeof(OneSweepSharedData<(1 << __radix_bits), __workgroup_size / SUBGROUP_SIZE, __data_per_workitem,
                                  __workgroup_size, oneapi::dpl::__internal::__value_t<_Range>>);
    const ::std::size_t __max_slm_size =
        __q.get_device().template get_info<sycl::info::device::local_mem_size>();
    assert(__req_slm_size <= __max_slm_size);

    // TODO: defer to parallel_backend_sycl_radix_sort_one_wg.h for small __n
    // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
    return __onesweep<_KernelName, __is_ascending, __radix_bits, __data_per_workitem, __workgroup_size>(
        __q, ::std::forward<_Range>(__rng), __n);
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_GPU_RADIX_SORT_DISPATCHERS_H
