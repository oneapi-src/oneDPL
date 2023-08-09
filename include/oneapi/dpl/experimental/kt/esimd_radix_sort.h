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

#include "kernel_param.h"
#include "esimd_radix_sort_one_wg.h"
#include "esimd_radix_sort_onesweep.h"

#include "../../pstl/utils_ranges.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include <cstdint>
#include <type_traits>

namespace oneapi::dpl::experimental::kt::esimd::impl
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

// TODO: allow calling it only for all_view (accessor) and guard_view (USM) ranges, views::subrange and sycl_iterator
template <bool _IsAscending, std::uint32_t _RadixBits, typename _KernelParam, typename _Range>
sycl::event
radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;

    const ::std::size_t __n = __rng.size();
    assert(__n > 1);

//    _PRINT_INFO_IN_DEBUG_MODE(__exec); TODO: extend the utility to work with queues
    using _KernelName = typename _KernelParam::kernel_name;

    if (__n <= 16384)
    {
        // TODO: support different RadixBits values (only 7 or 8 are currently supported), WorkGroupSize and DataPerWorkItem
        return oneapi::dpl::experimental::kt::esimd::impl::one_wg<_KernelName, _KeyT, _Range, _RadixBits, _IsAscending>(
            __q, ::std::forward<_Range>(__rng), __n);
    }
    else
    {
        // TODO: avoid kernel duplication (generate the output storage with the same type as input storage and use swap)
        // TODO: support different RadixBits, WorkGroupSize and DataPerWorkItem
        return oneapi::dpl::experimental::kt::esimd::impl::onesweep<_KernelName, _KeyT, _Range, _RadixBits, _IsAscending, __onesweep_process_size<_KeyT>()>(
            __q, ::std::forward<_Range>(__rng), __n);
    }
}

} // oneapi::dpl::experimental::kt::esimd::impl

namespace oneapi::dpl::experimental::kt::esimd
{

template <bool _IsAscending = true, std::uint32_t _RadixBits = 8, typename _KernelParam, typename _Range>
sycl::event
radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param = {})
{
    if(__rng.size() < 2)
        return {};

    return oneapi::dpl::experimental::kt::esimd::impl::radix_sort<_IsAscending, _RadixBits>(__q, ::std::forward<_Range>(__rng), __param);
}

template <bool _IsAscending = true, std::uint32_t _RadixBits = 8, typename _KernelParam, typename _Iterator>
sycl::event
radix_sort(sycl::queue __q, _Iterator __first, _Iterator __last, _KernelParam __param = {})
{
    if (__last - __first < 2)
        return {};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __rng = __keep(__first, __last);
    return oneapi::dpl::experimental::kt::esimd::impl::radix_sort<_IsAscending, _RadixBits>(__q, __rng.all_view(), __param);
}

} // namespace oneapi::dpl::experimental::kt::esimd

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
