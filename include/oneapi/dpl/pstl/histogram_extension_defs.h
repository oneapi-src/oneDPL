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

#ifndef _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
#define _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H

#include "onedpl_config.h"

namespace oneapi
{
namespace dpl
{
template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          typename std::iterator_traits<_RandomAccessIterator1>::value_type first_bin_min_val,
          typename std::iterator_traits<_RandomAccessIterator1>::value_type last_bin_max_val,
          _RandomAccessIterator2 histogram_first);

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator3>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last,
          _RandomAccessIterator2 boundary_first, _RandomAccessIterator2 boundary_last,
          _RandomAccessIterator3 histogram_first);

// Support extension API to cover existing API (previous to specification) with the following two overloads

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
          _ValueType first_bin_min_val, _ValueType last_bin_max_val, _RandomAccessIterator2 histogram_first);

// This overload is provided to support an extension to the oneDPL specification to support the original implementation
// of the histogram API, where if users explicitly-specify all template arguments, their arguments for bin boundary min
// and max are convertible to the specified _ValueType, and the _ValueType is convertible to the value type of the
// input iterator. Note that _ValueType is not deducable from the function arugments, so this overload is only used
// when users explicitly specify the _ValueType template argument.
template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2,
          typename _ValueType, typename _RealValueType1, typename _RealValueType2>
std::enable_if_t<
    oneapi::dpl::execution::is_execution_policy_v<std::decay_t<_ExecutionPolicy>> &&
        std::is_convertible_v<_ValueType, typename std::iterator_traits<_RandomAccessIterator1>::value_type> &&
        std::is_convertible_v<_RealValueType1, _ValueType> && std::is_convertible_v<_RealValueType2, _ValueType>,
    _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          _RealValueType1 first_bin_min_val, _RealValueType2 last_bin_max_val, _RandomAccessIterator2 histogram_first);

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_EXTENSION_DEFS_H
