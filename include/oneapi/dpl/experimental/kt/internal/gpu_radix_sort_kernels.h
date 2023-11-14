// -*- C++ -*-
//===-- gpu_radix_sort_kernels.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

template <typename _KeyT, typename _InputT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count,
          ::std::uint32_t __hist_work_group_count, ::std::uint32_t __hist_work_group_size, bool __is_ascending>
void
__global_histogram(sycl::nd_item<1> __idx, size_t __n, const _InputT& __input, ::std::uint32_t* __p_global_offset);

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _InputT, typename _OutputT>
struct __radix_sort_onesweep_slm_reorder_kernel;

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_GPU_RADIX_SORT_KERNELS_H
