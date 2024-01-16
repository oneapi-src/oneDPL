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

#ifndef _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
#define _ONEDPL_HISTOGRAM_BINHASH_UTILS_H

#include "utils_ranges.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
template <typename _T1, typename = void>
struct __evenly_divided_binhash;

template <typename _T1>
struct __evenly_divided_binhash<_T1, ::std::enable_if_t<::std::is_floating_point_v<_T1>>>
{
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;

    __evenly_divided_binhash(const _T1& __min, const _T1& __max, ::std::uint32_t __num_bins)
        : __minimum(__min), __maximum(__max), __scale(_T1(__num_bins) / (__max - __min))
    {
    }

    template <typename _T2>
    ::std::int32_t
    get_bin(const _T2& __value) const
    {
        int ret = -1; 
        if ((__value >= __minimum) && (__value < __maximum))
        {
            ret = (__value - __minimum) * __scale;
        }
        return ret;
    }
};

template <typename _T1>
struct __evenly_divided_binhash<_T1, ::std::enable_if_t<!::std::is_floating_point_v<_T1>>>
{
    _T1 __minimum;
    _T1 __range_size;
    ::std::uint32_t __num_bins;
    __evenly_divided_binhash(const _T1& __min, const _T1& __max, ::std::uint32_t __num_bins_)
        : __minimum(__min), __num_bins(__num_bins_), __range_size(__max - __min)
    {
    }

    template <typename _T2>
    ::std::int32_t 
    get_bin(const _T2& __value) const
    {
        int ret = -1;
        if ((__value >= __minimum) && (__value < (__minimum + __range_size)))
        {
            ret = ((::std::uint64_t(__value) - __minimum) * ::std::uint64_t(__num_bins)) / __range_size;
        }
        return ret;
    }
};

template <typename _Range>
struct __custom_range_binhash
{
    using _range_value_type = oneapi::dpl::__internal::__value_t<_Range>;
    _Range __boundaries;

    __custom_range_binhash(_Range __boundaries_) : __boundaries(__boundaries_) {}

    template <typename _BoundaryIter, typename _T2, typename _T3>
    static int
    get_bin_helper(_BoundaryIter __first, _BoundaryIter __last, _T2 __value, _T3 __min, _T3 __max)
    {
        int ret = -1;
        if (__value >= __min && __value < __max)
        {
            ret = std::distance(__first, ::std::upper_bound(__first, __last, ::std::forward<_T2>(__value))) - 1;
        }
        return ret;
    }

    template <typename _T2>
    ::std::int32_t
    get_bin(_T2&& __value) const
    {
        return get_bin_helper(__boundaries.begin(), __boundaries.end(), ::std::forward<_T2>(__value), __boundaries[0], __boundaries[__boundaries.size()-1]);
    }

    _Range
    get_range() const
    {
        return __boundaries;
    }
};

} // end namespace __internal
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
