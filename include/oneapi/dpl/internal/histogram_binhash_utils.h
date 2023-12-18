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

#include "../pstl/utils_ranges.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
template <typename _T1, bool _IsFloatingPoint>
struct __evenly_divided_binhash_impl
{
};

template <typename _T1>
struct __evenly_divided_binhash_impl<_T1, /* _IsFloatingPoint = */ true>
{
    //Does not require creation of any SYCL range in hetero backend
    using req_sycl_range_conversion = ::std::false_type;
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;

    __evenly_divided_binhash_impl(const _T1& __min, const _T1& __max, ::std::uint32_t __num_bins)
        : __minimum(__min), __maximum(__max), __scale(_T1(__num_bins) / (__max - __min))
    {
    }

    template <typename _T2>
    ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return ::std::uint32_t((::std::forward<_T2>(__value) - __minimum) * __scale);
    }

    template <typename _T2>
    bool
    is_valid(const _T2& __value) const
    {
        return (__value >= __minimum) && (__value < __maximum);
    }
};

template <typename _T1>
struct __evenly_divided_binhash_impl<_T1, /* _IsFloatingPoint= */ false>
{
    //Does not require creation of any SYCL range in hetero backend
    using req_sycl_range_conversion = ::std::false_type;
    _T1 __minimum;
    _T1 __range_size;
    ::std::uint32_t __num_bins;
    __evenly_divided_binhash_impl(const _T1& __min, const _T1& __max, ::std::uint32_t __num_bins_)
        : __minimum(__min), __num_bins(__num_bins_), __range_size(__max - __min)
    {
    }
    template <typename _T2>
    ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return ::std::uint32_t(
            ((::std::uint64_t(::std::forward<_T2>(__value)) - __minimum) * ::std::uint64_t(__num_bins)) / __range_size);
    }

    template <typename _T2>
    bool
    is_valid(const _T2& __value) const
    {
        return (__value >= __minimum) && (__value < (__minimum + __range_size));
    }
};

template <typename _T1>
using __evenly_divided_binhash = __evenly_divided_binhash_impl<_T1, std::is_floating_point_v<_T1>>;

template <typename _Range>
struct __custom_range_binhash
{
    //Requires creation of __boundaries SYCL range if used in hetero backend
    using req_sycl_range_conversion = ::std::true_type;
    using __boundary_type = oneapi::dpl::__internal::__value_t<_Range>;
    _Range __boundaries;

    __custom_range_binhash(_Range __boundaries_) : __boundaries(__boundaries_) {}

    template <typename _T2>
    ::std::uint32_t
    get_bin(_T2&& __value) const
    {
        return std::distance(__boundaries.begin(), ::std::upper_bound(__boundaries.begin(), __boundaries.end(),
                                                                      ::std::forward<_T2>(__value))) -
               1;
    }

    template <typename _T2>
    bool
    is_valid(const _T2& __value) const
    {
        return __value >= __boundaries[0] && __value < __boundaries[__boundaries.size() - 1];
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
