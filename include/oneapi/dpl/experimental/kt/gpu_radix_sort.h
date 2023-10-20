// -*- C++ -*-
//===-- gpu_radix_sort.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_GPU_RADIX_SORT_H
#define _ONEDPL_KT_GPU_RADIX_SORT_H

#include "kernel_param.h"
#include "../../pstl/utils_ranges.h"
#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include <cstdint>

#include "internal/gpu_radix_sort_dispatchers.h"

namespace oneapi::dpl::experimental::kt::gpu
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

} // namespace oneapi::dpl::experimental::kt::gpu

#endif // _ONEDPL_KT_GPU_RADIX_SORT_H
