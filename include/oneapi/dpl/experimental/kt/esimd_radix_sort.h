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

#include "kernel_param.h"
#include "esimd_radix_sort_dispatchers.h"
#include "../../pstl/utils_ranges.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include <cstdint>

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

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
