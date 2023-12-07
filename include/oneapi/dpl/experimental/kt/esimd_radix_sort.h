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

#include <cstdint>

#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "internal/esimd_radix_sort_utils.h"
#include "internal/esimd_radix_sort_dispatchers.h"

namespace oneapi::dpl::experimental::kt::esimd
{

// TODO: make sure to provide sufficient diagnostic if input does not allow either reading or writing
template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng>
sycl::event
radix_sort(sycl::queue __q, _KeysRng&& __keys_rng, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng>(__keys_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, ::std::move(__pack), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator>
sycl::event
radix_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_last - __keys_first < 2)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, ::std::move(__pack), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng, typename _ValsRng>
sycl::event
radix_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng>(__keys_rng)),
                                              oneapi::dpl::__ranges::views::all(::std::forward<_ValsRng>(__vals_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, ::std::move(__pack), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator, typename _ValsIterator>
sycl::event
radix_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _ValsIterator __vals_first, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_last - __keys_first < 2)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();

    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _ValsIterator>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + (__keys_last - __keys_first)).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng), ::std::move(__vals_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits>(__q, ::std::move(__pack), __param);
}

} // namespace oneapi::dpl::experimental::kt::esimd

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
