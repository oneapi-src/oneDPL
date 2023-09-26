// -*- C++ -*-
//===-- esimd_radix_sort_by_key.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_BY_KEY_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_BY_KEY_H

#include <ext/intel/esimd.hpp>

#include "kernel_param.h"
#include "esimd_radix_sort_onesweep_by_key.h"

#include "../../pstl/utils_ranges.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include <cstdint>
#include <type_traits>

namespace oneapi::dpl::experimental::kt::esimd::__impl
{

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

namespace oneapi::dpl::experimental::kt::esimd
{

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng, typename _ValsRng>
sycl::event
radix_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, _KernelParam __param = {})
{
    if (__keys_rng.size() < 2)
        return {};

    return __impl::__radix_sort_by_key<__is_ascending, __radix_bits>(__q, ::std::forward<_KeysRng>(__keys_rng), ::std::forward<_ValsRng>(__vals_rng), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator, typename _ValsIterator>
sycl::event
radix_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _ValsIterator __vals_first, _KernelParam __param = {})
{
    if (__keys_last - __keys_first < 2)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last);

    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write,
                                                          _ValsIterator>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + (__keys_last - __keys_first));

    return __impl::__radix_sort_by_key<__is_ascending, __radix_bits>(__q, __keys_rng.all_view(), __vals_rng.all_view(), __param);
}

} // namespace oneapi::dpl::experimental::kt::esimd

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_BY_KEY_H
