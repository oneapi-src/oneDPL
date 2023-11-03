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

#ifndef _HISTOGRAM_SERIAL_IMPL_H
#define _HISTOGRAM_SERIAL_IMPL_H

template <typename _T1, bool _IsFloatingPoint>
struct evenly_divided_binhash_impl{};

template <typename _T>
struct evenly_divided_binhash_impl<_T, /* is_floating_point = */ true>
{
    _T __minimum;
    _T __scale;
    evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::size_t& num_bins)
        : __minimum(min), __scale(_T(num_bins) / (max - min))
    {
    }
    template <typename _T2>
    ::std::uint32_t
    operator()(_T2&& value) const
    {
        return ::std::uint32_t((::std::forward<_T2>(value) - __minimum) * __scale);
    }

};

template <typename _T>
struct evenly_divided_binhash_impl<_T, /* is_floating_point= */ false>
{
    _T __minimum;
    ::std::size_t __num_bins;
    _T __range_size;
    evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::size_t& num_bins)
        : __minimum(min), __num_bins(num_bins), __range_size(max - min)
    {
    }
    template <typename _T2>
    ::std::uint32_t
    operator()(_T2&& value) const
    {
        return ::std::uint32_t(((_T(::std::forward<_T2>(value)) - __minimum) * _T(__num_bins)) / __range_size);
    }

};

template <typename _T1>
using evenly_divided_binhash = evenly_divided_binhash_impl<_T1, std::is_floating_point_v<_T1>>;

template <typename _ForwardIterator>
struct custom_range_binhash
{
    _ForwardIterator __first;
    _ForwardIterator __last;
    custom_range_binhash(_ForwardIterator first, _ForwardIterator last)
        : __first(first), __last(last)
    {
    }

    template <typename _T>
    ::std::uint32_t
    operator()(_T&& value) const
    {
        return (::std::upper_bound(__first, __last, ::std::forward<_T>(value)) - __first) - 1;
    }

};

template <typename _ForwardIterator, typename _RandomAccessIterator, typename _Size, typename _IdxHashFunc>
_RandomAccessIterator
histogram_general_sequential(_ForwardIterator __first, _ForwardIterator __last, _RandomAccessIterator __histogram_first,
                  _Size num_bins, _IdxHashFunc __func)
{
    for (int bin = 0; bin < num_bins; bin++)
    {
        __histogram_first[bin] = 0;
    }

    for (auto tmp = __first; tmp < __last; ++tmp)
    {
        _Size bin = __func(*tmp);
        if (bin >= 0 && bin < num_bins)
            ++(__histogram_first[bin]);
    }
    return __histogram_first + num_bins;
}


template <typename _InputIterator1, typename _Size, typename _T, typename _OutputIterator>
_OutputIterator
histogram_sequential(_InputIterator1 __first, _InputIterator1 __last, const _Size& __num_bins,
                     const _T& __first_bin_min_val, const _T& __last_bin_max_val, _OutputIterator __histogram_first)
{
    return histogram_general_sequential(__first, __last, __histogram_first, __num_bins,
                             evenly_divided_binhash<_T>(__first_bin_min_val, __last_bin_max_val, __num_bins));
}


template <typename _InputIterator1, typename _InputIterator2, typename _OutputIterator>
_OutputIterator
histogram_sequential(_InputIterator1 __first, _InputIterator1 __last, _InputIterator2 __boundary_first,
                     _InputIterator2 __boundary_last, _OutputIterator __histogram_first)
{
    return histogram_general_sequential(__first, __last, __histogram_first, (__boundary_last - __boundary_first) - 1,
                             custom_range_binhash{__boundary_first, __boundary_last});
}



#endif // _HISTOGRAM_SERIAL_IMPL_H 
