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

namespace oneapi::dpl::experimental::kt::esimd::__impl
{

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

} // namespace oneapi::dpl::experimental::kt::esimd::__impl

namespace oneapi::dpl::experimental::kt::esimd
{

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _Range>
sycl::event
radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param = {})
{
    if (__rng.size() < 2)
        return {};

    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, ::std::forward<_Range>(__rng), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _Iterator>
sycl::event
radix_sort(sycl::queue __q, _Iterator __first, _Iterator __last, _KernelParam __param = {})
{
    if (__last - __first < 2)
        return {};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _Iterator>();
    auto __rng = __keep(__first, __last);
    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, __rng.all_view(), __param);
}

} // namespace oneapi::dpl::experimental::kt::esimd

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
