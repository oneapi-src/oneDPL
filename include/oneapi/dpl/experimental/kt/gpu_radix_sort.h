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

namespace oneapi::dpl::experimental::kt::gpu
{

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _Range>
sycl::event
radix_sort(sycl::queue __q, _Range&& __rng, _KernelParam __param = {});

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _Iterator>
sycl::event
radix_sort(sycl::queue __q, _Iterator __first, _Iterator __last, _KernelParam __param = {});

} // namespace oneapi::dpl::experimental::kt::gpu

#endif // _ONEDPL_KT_GPU_RADIX_SORT_H
