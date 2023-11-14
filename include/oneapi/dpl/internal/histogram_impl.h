// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _ONEDPL_HISTOGRAM_IMPL_H
#define _ONEDPL_HISTOGRAM_IMPL_H

#include "function.h"
#include "histogram_extension_defs.h"
#include "histogram_binhash_utils.h"
#include "../pstl/iterator_impl.h"

#if _ONEDPL_HETERO_BACKEND
#    include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_histogram.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

#if _ONEDPL_BACKEND_SYCL

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Size,
          typename _IdxHashFunc, typename... _Range>
inline void
__pattern_histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                    _RandomAccessIterator2 __histogram_first, const _Size& __num_bins, _IdxHashFunc __func,
                    _Range&&... __opt_range)
{
    //If there are no histogram bins there is nothing to do
    if (__num_bins > 0)
    {
        oneapi::dpl::__par_backend_hetero::__parallel_histogram(::std::forward<_ExecutionPolicy>(exec), __first, __last,
                                                                __histogram_first, __num_bins, __func, __opt_range...);
    }
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Size,
          typename _T>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<typename ::std::decay<_ExecutionPolicy>::type>
__histogram_impl(_ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                 _RandomAccessIterator2 __histogram_first, const _Size& __num_bins, const _T& __first_bin_min_val,
                 const _T& __last_bin_max_val)
{
    __internal::__pattern_histogram(
        ::std::forward<_ExecutionPolicy>(exec), __first, __last, __histogram_first, __num_bins,
        __internal::__evenly_divided_binhash<_T>(__first_bin_min_val, __last_bin_max_val, __num_bins));
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename Iter3>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<typename ::std::decay<_ExecutionPolicy>::type>
__histogram_impl(_ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                 _RandomAccessIterator2 __histogram_first, Iter3 __boundary_first, Iter3 __boundary_last)
{
    auto boundary_view = oneapi::dpl::__ranges::guard_view<Iter3>(__boundary_first, __boundary_last);
    __internal::__pattern_histogram(::std::forward<_ExecutionPolicy>(exec), __first, __last, __histogram_first,
                                    boundary_view.size() - 1, __internal::__custom_range_binhash{boundary_view});
}

#endif // _ONEDPL_BACKEND_SYCL

} // namespace __internal

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _T,
          typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
          const _Size& __num_bins, const _T& __first_bin_min_val, const _T& __last_bin_max_val,
          _RandomAccessIterator2 __histogram_first)
{
    __internal::__histogram_impl(::std::forward<_ExecutionPolicy>(exec), __first, __last, __histogram_first, __num_bins,
                                 __first_bin_min_val, __last_bin_max_val);
    return __histogram_first + __num_bins;
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator3>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
          _RandomAccessIterator2 __boundary_first, _RandomAccessIterator2 __boundary_last,
          _RandomAccessIterator3 __histogram_first)
{
    ::std::ptrdiff_t __num_bins = __boundary_last - __boundary_first - 1;
    __internal::__histogram_impl(::std::forward<_ExecutionPolicy>(exec), __first, __last, __histogram_first,
                                 __boundary_first, __boundary_last);
    return __histogram_first + __num_bins;
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_IMPL_H
