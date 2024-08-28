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

#include "histogram_extension_defs.h"
#include "histogram_binhash_utils.h"
#include "iterator_impl.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/histogram_impl_hetero.h"
#endif

namespace oneapi
{
namespace dpl
{

namespace __internal
{

template <class _Tag, typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _IdxHashFunc,
          typename _RandomAccessIterator2>
void
__pattern_histogram(_Tag, _ExecutionPolicy&& exec, _RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                    _Size __num_bins, _IdxHashFunc __func, _RandomAccessIterator2 __histogram_first)
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    static_assert(sizeof(_Size) == 0 /*false*/,
                  "Histogram API is currently unsupported for policies other than device execution policies");
}

} // namespace __internal

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          typename std::iterator_traits<_RandomAccessIterator1>::value_type first_bin_min_val,
          typename std::iterator_traits<_RandomAccessIterator1>::value_type last_bin_max_val,
          _RandomAccessIterator2 histogram_first)
{
    using _BoundaryType = typename std::iterator_traits<_RandomAccessIterator1>::value_type;
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, histogram_first);

    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__evenly_divided_binhash<_BoundaryType>(first_bin_min_val, last_bin_max_val, num_bins),
        histogram_first);
    return histogram_first + num_bins;
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator3>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last,
          _RandomAccessIterator2 boundary_first, _RandomAccessIterator2 boundary_last,
          _RandomAccessIterator3 histogram_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, boundary_first, histogram_first);

    ::std::ptrdiff_t num_bins = boundary_last - boundary_first - 1;
    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__custom_boundary_binhash{boundary_first, boundary_last}, histogram_first);
    return histogram_first + num_bins;
}

// Support extention API to cover existing API (previous to specification) with the following two overloads

// This overload is provided to support an extension to the oneDPL specification to support the original implementation
// of the histogram API, where the boundary type _ValueType could differ from the value type of the input iterator,
// and required `<`, `<=`, `+`, `-`, and `/` to be defined between _ValueType and
// std::iterator_traits<_RandomAccessIterator1>::value_type rather than enforcing they were the same type or convertible
template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2,
          typename _ValueType>
std::enable_if_t<
    oneapi::dpl::execution::is_execution_policy_v<std::decay_t<_ExecutionPolicy>> &&
        !std::is_convertible_v<_ValueType, typename std::iterator_traits<_RandomAccessIterator1>::value_type>,
    _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          _ValueType first_bin_min_val, _ValueType last_bin_max_val, _RandomAccessIterator2 histogram_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, histogram_first);

    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__evenly_divided_binhash<_ValueType>(first_bin_min_val, last_bin_max_val, num_bins),
        histogram_first);
    return histogram_first + num_bins;
}

// This overload is provided to support an extension to the oneDPL specification to support the original implementation
// of the histogram API, where if users explicitly-specify all template arguments, their arguments for bin boundary min
// and max are convertible to the specified _ValueType, and the _ValueType is convertible to the value type of the
// input iterator.
template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2,
          typename _ValueType, typename _RealValueType1, typename _RealValueType2>
std::enable_if_t<
    oneapi::dpl::execution::is_execution_policy_v<std::decay_t<_ExecutionPolicy>> &&
        std::is_convertible_v<_ValueType, typename std::iterator_traits<_RandomAccessIterator1>::value_type> &&
        std::is_convertible_v<_RealValueType1, _ValueType> && std::is_convertible_v<_RealValueType2, _ValueType>,
    _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          _RealValueType1 first_bin_min_val, _RealValueType2 last_bin_max_val, _RandomAccessIterator2 histogram_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, histogram_first);

    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__evenly_divided_binhash<_ValueType>(_ValueType{first_bin_min_val},
                                                                      _ValueType{last_bin_max_val}, num_bins),
        histogram_first);
    return histogram_first + num_bins;
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_IMPL_H
