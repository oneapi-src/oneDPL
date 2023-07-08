// -*- C++ -*-
//===-- esimd_radix_sort.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_H

#include <ext/intel/esimd.hpp>

#include "esimd_radix_sort_one_wg.h"
#include "esimd_radix_sort_cooperative.h"
#include "esimd_radix_sort_onesweep.h"
#include "esimd_radix_sort_onesweep_by_key.h"

#include "../../pstl/utils_ranges.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../pstl/hetero/dpcpp/execution_sycl_defs.h"

#include <cstdint>
#include <type_traits>

namespace oneapi::dpl::experimental::esimd::impl
{
template <typename T, ::std::enable_if_t<sizeof(T) <= sizeof(::std::uint32_t), int> = 0>
constexpr ::std::uint32_t
__onesweep_process_size()
{
    return 384;
}

template <typename T, ::std::enable_if_t<sizeof(T) == sizeof(::std::uint64_t), int> = 0>
constexpr ::std::uint32_t
__onesweep_process_size()
{
    return 192;
}

/*
    Interface:
        Provide a temporary buffer. May be not necessary for some implementations, e.g. one-work-group implementation.
            Provide a way to show required size of a temporary memory pool.
        Sort only specific range of bits?
            Add a tag for implementation with requiring forward progress
            Allow selection of specific implementation?
*/
/*
     limitations:
         eSIMD operations:
            gather: Element type; can only be a 1,2,4-byte integer, sycl::half or float.
            lsc_gather: limited supported platforms: see https://intel.github.io/llvm-docs/doxygen/group__sycl__esimd__memory__lsc.html#ga250b3c0085f39c236582352fb711aadb)
*/
// TODO: call it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending, std::uint32_t RadixBits,
          typename _ExecutionPolicy, typename _Range,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
void
radix_sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;

    const ::std::size_t __n = __rng.size();
    assert(__n > 1);

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    using _KernelName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;

    if (__n <= 16384)
    {
        // TODO: support different RadixBits values (only 7 or 8 are currently supported), WorkGroupSize and DataPerWorkItem
        oneapi::dpl::experimental::esimd::impl::one_wg<_KernelName, _KeyT, _Range, RadixBits, IsAscending>(
            __exec.queue(), ::std::forward<_Range>(__rng), __n);
    }
    else if (__n <= 262144)
    {
        // TODO: support different RadixBits, WorkGroupSize and DataPerWorkItem
        oneapi::dpl::experimental::esimd::impl::cooperative<_KernelName, _KeyT, _Range, RadixBits, IsAscending>(
            __exec.queue(), ::std::forward<_Range>(__rng), __n);
    }
    else
    {
        // TODO: enable support of double type
        // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
        // TODO: pass _ProcessSize according to DataPerWorkItem
        // TODO: support different WorkGroupSize, RadixBits
        oneapi::dpl::experimental::esimd::impl::onesweep<_KernelName, _KeyT, _Range, RadixBits, IsAscending, __onesweep_process_size<_KeyT>()>(
            __exec.queue(), ::std::forward<_Range>(__rng), __n);
    }
}

template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending, std::uint32_t RadixBits,
          typename _ExecutionPolicy, typename _KeysRange, typename _ValuesRange,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_ExecutionPolicy, int> = 0>
void
radix_sort_by_key(_ExecutionPolicy&& __exec, _KeysRange&& __keys, _ValuesRange&& __values)
{
    const ::std::size_t __n = __keys.size();
    assert(__n > 1);

    _PRINT_INFO_IN_DEBUG_MODE(__exec);
    using _KernelName = typename ::std::decay_t<_ExecutionPolicy>::kernel_name;
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRange>;
    using _ValueT = oneapi::dpl::__internal::__value_t<_ValuesRange>;

    oneapi::dpl::experimental::esimd::impl::onesweep_by_key<_KernelName, _KeyT, _ValueT, _KeysRange, _ValuesRange, RadixBits, IsAscending, DataPerWorkItem>(
        __exec.queue(), ::std::forward<_KeysRange>(__keys), std::forward<_ValuesRange>(__values), __n);
}

} // oneapi::dpl::experimental::esimd::impl

namespace oneapi::dpl::experimental::esimd
{
template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending = true,
          std::uint32_t RadixBits = 8, typename _ExecutionPolicy, typename _Range>
void
radix_sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    if(__rng.size() < 2)
        return;
    oneapi::dpl::experimental::esimd::impl::radix_sort<WorkGroupSize, DataPerWorkItem, IsAscending, RadixBits>
        (::std::forward<_ExecutionPolicy>(__exec), __rng);
}

template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending = true,
          std::uint32_t RadixBits = 8, typename _ExecutionPolicy, typename _Iterator>
void
radix_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last)
{
    if (__last - __first < 2)
        return;
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __rng = __keep(__first, __last);
    oneapi::dpl::experimental::esimd::impl::radix_sort<WorkGroupSize, DataPerWorkItem, IsAscending, RadixBits>
        (::std::forward<_ExecutionPolicy>(__exec), __rng.all_view());
}

template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending = true,
          std::uint32_t RadixBits = 8, typename _ExecutionPolicy, typename _KeysRange, typename _ValuesRange>
void
radix_sort_by_key(_ExecutionPolicy&& __exec, _KeysRange&& __keys, _ValuesRange&& __values)
{
    if (__keys.size() < 2)
        return;

    oneapi::dpl::experimental::esimd::impl::radix_sort_by_key<WorkGroupSize, DataPerWorkItem, IsAscending, RadixBits>
        (::std::forward<_ExecutionPolicy>(__exec), __keys, __values);
}

template <std::uint16_t WorkGroupSize, std::uint16_t DataPerWorkItem, bool IsAscending = true,
          std::uint32_t RadixBits = 8, typename _ExecutionPolicy, typename _KeysIterator, typename _ValuesIterator>
void
radix_sort_by_key(_ExecutionPolicy&& __exec, _KeysIterator __keys_first, _KeysIterator __keys_last, _ValuesIterator __values_first)
{
    auto __n = __keys_last - __keys_first;
    if (__n < 2)
        return;
    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _KeysIterator>();
    auto __keys = __keys_keep(__keys_first, __keys_last);
    auto __values_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _ValuesIterator>();
    auto __values = __values_keep(__values_first, __values_first + __n);
    oneapi::dpl::experimental::esimd::impl::radix_sort_by_key<WorkGroupSize, DataPerWorkItem, IsAscending, RadixBits>
        (::std::forward<_ExecutionPolicy>(__exec), __keys.all_view(), __values.all_view());
}

} // namespace oneapi::dpl::experimental::esimd

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
