// -*- C++ -*-
//===-- parallel_backend_esimd_radix_sort.h --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_esimd_radix_sort_H
#define _ONEDPL_parallel_backend_esimd_radix_sort_H

#include <ext/intel/esimd.hpp>
#include "sycl_defs.h"

#include "esimd_radix_sort/esimd_radix_sort_one_wg.h"
#include "esimd_radix_sort/esimd_radix_sort_cooperative.h"
#include "esimd_radix_sort/esimd_radix_sort_onesweep.h"

#include "../../utils.h"
#include "../../utils_ranges.h"
#include "utils_ranges_sycl.h"

#include <cstdint>
#include <type_traits>

namespace oneapi::dpl::experimental::esimd
{
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
// TODO: call it only for all_view (accessor) and guard_view (USM) ranges and probably subrange (according to the documentation, not tested)
template <typename _ExecutionPolicy, typename _Range, bool IsAscending = true, std::uint16_t WorkGroupSize = 256,
          std::uint16_t ItemsPerWorkItem = 16, std::uint32_t RadixBits = 8>
void
radix_sort(_ExecutionPolicy&& __exec, _Range&& __rng)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_Range>;

    auto __q = __exec.queue();
    const ::std::size_t __n = __rng.size();

    if (__n <= 16384)
    {
        // TODO: allow differnt sorting orders
        // TODO: allow diferent types
        // TODO: generate unique kernel names
        // TODO: allow all RadixBits values (only 7 or 8 are currently supported)
        oneapi::dpl::experimental::esimd::impl::one_wg<_KeyT, _Range, RadixBits>(__q, ::std::forward<_Range>(__rng), __n);
    }
    else if (__n <= 262144)
    {
        // TODO: allow differnt sorting orders
        // TODO: allow diferent types
        // TODO: generate unique kernel names
        oneapi::dpl::experimental::esimd::impl::cooperative<_KeyT, _Range, RadixBits>(__q, ::std::forward<_Range>(__rng), __n);
    }
    else
    {
        // TODO: allow differnt sorting orders
        // TODO: allow diferent types
        // TODO: generate unique kernel names
        // TODO: avoid kernel duplication (generate the output storate with the same type as __data and use swap)
        // TODO: allow different RadixBits, make sure the data is in __data after the last stage
        // TODO: pass process_size according to __n
        oneapi::dpl::experimental::esimd::impl::onesweep<_KeyT, _Range, RadixBits>(__q, ::std::forward<_Range>(__rng), __n, /*process_size*/ 512);
    }
}

template <typename _ExecutionPolicy, typename _Iterator, bool IsAscending = true, std::uint16_t WorkGroupSize = 256,
          std::uint16_t ItemsPerWorkItem = 16, std::uint32_t RadixBits = 8>
void
radix_sort(_ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __rng = __keep(__first, __last);
    radix_sort(::std::forward<_ExecutionPolicy>(__exec), __rng.all_view());
}

} // namespace oneapi::dpl::experimental::esimd

#endif // _ONEDPL_parallel_backend_esimd_radix_sort_H
