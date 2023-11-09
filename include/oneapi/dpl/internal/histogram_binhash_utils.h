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
struct __evenly_divided_binhash_impl<_T1, /* is_floating_point = */ true>
{
    using req_sycl_range_conversion = ::std::false_type;
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;

    __evenly_divided_binhash_impl(const _T1& min, const _T1& max, const ::std::uint32_t& num_bins)
        : __minimum(min), __maximum(max), __scale(_T1(num_bins) / (max - min))
    {
    }

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value) const
    {
        return ::std::uint32_t((::std::forward<_T2>(value) - __minimum) * __scale);
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value) const
    {
        return (value >= __minimum) && (value < __maximum);
    }
};

// non floating point type
template <typename _T1>
struct __evenly_divided_binhash_impl<_T1, /* is_floating_point= */ false>
{
    using req_sycl_range_conversion = ::std::false_type;
    _T1 __minimum;
    _T1 __range_size;
    ::std::uint32_t __num_bins;
    __evenly_divided_binhash_impl(const _T1& min, const _T1& max, const ::std::uint32_t& num_bins)
        : __minimum(min), __num_bins(num_bins), __range_size(max - min)
    {
    }
    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value) const
    {
        return ::std::uint32_t(
            ((::std::uint64_t(::std::forward<_T2>(value)) - __minimum) * ::std::uint64_t(__num_bins)) / __range_size);
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value) const
    {
        return (value >= __minimum) && (value < (__minimum + __range_size));
    }
};

template <typename _T1>
using __evenly_divided_binhash = __evenly_divided_binhash_impl<_T1, std::is_floating_point_v<_T1>>;

template <typename _Range>
struct __custom_range_binhash
{
    using req_sycl_range_conversion = ::std::true_type;
    using __boundary_type = oneapi::dpl::__internal::__value_t<_Range>;
    _Range __boundaries;

    __custom_range_binhash(_Range boundaries) : __boundaries(boundaries) {}

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value) const
    {
        return (::std::upper_bound(__boundaries.begin(), __boundaries.end(), ::std::forward<_T>(value)) -
                __boundaries.begin()) -
               1;
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value) const
    {
        return value >= __boundaries[0] && value < __boundaries[__boundaries.size() - 1];
    }

    _Range
    get_range()
    {
        return __boundaries;
    }
};

} // end namespace __internal
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
