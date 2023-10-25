// -*- C++ -*-
//===-- gpu_radix_sort_submitters.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_GPU_RADIX_SORT_SUBMITTERS_H
#define _ONEDPL_KT_GPU_RADIX_SORT_SUBMITTERS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include <cstdint>

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//------------------------------------------------------------------------
// Please see the comment for __parallel_for_submitter for optional kernel name explanation
//------------------------------------------------------------------------

template <typename _KeyT, ::std::uint32_t __radix_bits, ::std::uint32_t __stage_count,
          ::std::uint32_t __hist_work_group_count, ::std::uint32_t __hist_work_group_size, bool __is_ascending,
          typename _KernelName>
struct __radix_sort_onesweep_histogram_submitter;

template <::std::uint32_t __stage_count, ::std::uint32_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <typename _KeyT, typename _KernelName>
struct __radix_sort_copyback_submitter;

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_GPU_RADIX_SORT_SUBMITTERS_H
