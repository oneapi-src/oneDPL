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
#include "../pstl/iterator_impl.h"
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_histogram.h"

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename _T1, bool _IsFloatingPoint>
struct __evenly_divided_binhash_impl
{
};

template <typename _T>
struct __evenly_divided_binhash_impl<_T, /* is_floating_point = */ true>
{
    _T __minimum;
    _T __maximum;
    _T __scale;

    __evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::uint32_t& num_bins)
        : __minimum(min), __maximum(max), __scale(_T(num_bins) / (max - min))
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

#if _ONEDPL_BACKEND_SYCL

    inline ::std::size_t
    get_required_SLM_memory() const
    {
        return 0;
    }

    inline void
    init_SLM_memory(void* boost_mem, const sycl::nd_item<1>& self_item) const
    {
    }

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value, void* boost_mem) const
    {
        return get_bin(::std::forward<_T2>(value));
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value, void* boost_mem) const
    {
        return is_valid(::std::forward<_T2>(value));
    }

#endif // _ONEDPL_BACKEND_SYCL
};

// non floating point type
template <typename _T>
struct __evenly_divided_binhash_impl<_T, /* is_floating_point= */ false>
{
    _T __minimum;
    _T __range_size;
    ::std::uint32_t __num_bins;
    __evenly_divided_binhash_impl(const _T& min, const _T& max, const ::std::uint32_t& num_bins)
        : __minimum(min), __num_bins(num_bins), __range_size(max - min)
    {
    }
    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value) const
    {
        return ::std::uint32_t(((::std::uint64_t(::std::forward<_T2>(value)) - __minimum) * ::std::uint64_t(__num_bins)) / __range_size);
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value) const
    {
        return (value >= __minimum) && (value < (__minimum + __range_size));
    }

#if _ONEDPL_BACKEND_SYCL

    inline ::std::size_t
    get_required_SLM_memory() const
    {
        return 0;
    }

    inline void
    init_SLM_memory(void* boost_mem, const sycl::nd_item<1>& self_item) const
    {
    }

    template <typename _T2>
    inline ::std::uint32_t
    get_bin(_T2&& value, void* boost_mem) const
    {
        return get_bin(::std::forward<_T2>(value));
    }

    template <typename _T2>
    inline bool
    is_valid(const _T2& value, void* boost_mem) const
    {
        return is_valid(::std::forward<_T2>(value));
    }

#endif // _ONEDPL_BACKEND_SYCL

};

template <typename _T1>
using __evenly_divided_binhash = __evenly_divided_binhash_impl<_T1, std::is_floating_point_v<_T1>>;

#if _ONEDPL_BACKEND_SYCL

template <typename _Range>
struct __custom_range_binhash
{
    using __boundary_type = oneapi::dpl::__internal::__value_t<_Range>;
    _Range __boundaries;
    __custom_range_binhash(_Range boundaries) : __boundaries(boundaries) {}

    template <typename _T>
    inline ::std::uint32_t
    get_bin(_T&& value) const
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


    inline ::std::size_t
    get_required_SLM_memory()
    {
        return sizeof(__boundary_type) * __boundaries.size();
    }

    inline void
    init_SLM_memory(void* boost_mem, const sycl::nd_item<1>& self_item) const
    {
        ::std::uint32_t gSize = self_item.get_local_range()[0];
        ::std::uint32_t self_lidx = self_item.get_local_id(0);
        ::std::uint8_t factor = oneapi::dpl::__internal::__dpl_ceiling_div(__boundaries.size(), gSize);
        ::std::uint8_t k;
        __boundary_type* d_boundaries = (__boundary_type*)(boost_mem);
        _ONEDPL_PRAGMA_UNROLL
        for (k = 0; k < factor - 1; k++)
        {
            d_boundaries[gSize * k + self_lidx] = __boundaries[gSize * k + self_lidx];
        }
        // residual
        if (gSize * k + self_lidx < __boundaries.size())
        {
            d_boundaries[gSize * k + self_lidx] = __boundaries[gSize * k + self_lidx];
        }
    }

    template <typename _T2>
    ::std::uint32_t inline get_bin(_T2&& value, void* boost_mem) const
    {
        __boundary_type* d_boundaries = (__boundary_type*)(boost_mem);
        return (::std::upper_bound(d_boundaries, d_boundaries + __boundaries.size(), ::std::forward<_T2>(value)) -
                d_boundaries) -
               1;
    }

    template <typename _T2>
    bool inline is_valid(const _T2& value, void* boost_mem) const
    {
        __boundary_type* d_boundaries = (__boundary_type*)(boost_mem);
        return (value >= d_boundaries[0]) && (value < d_boundaries[__boundaries.size()]);
    }

#endif // _ONEDPL_BACKEND_SYCL

};

template <typename Policy, typename _Iter1, typename _Iter2, typename _Size, typename _IdxHashFunc, typename... _Range>
inline _Iter2
__pattern_histogram(Policy&& policy, _Iter1 __first, _Iter1 __last, _Iter2 __histogram_first, const _Size& num_bins,
                    _IdxHashFunc __func, _Range&&... __opt_range)
{
    return oneapi::dpl::__par_backend_hetero::__parallel_histogram(::std::forward<Policy>(policy), __first, __last,
                                                                   __histogram_first, num_bins, __func, __opt_range...);
}

#if _ONEDPL_BACKEND_SYCL

template <typename Policy, typename Iter1, typename OutputIter, typename _Size, typename _T>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<typename ::std::decay<Policy>::type, OutputIter>
__histogram_impl(Policy&& policy, Iter1 __first, Iter1 __last, OutputIter __histogram_first, const _Size& num_bins,
                 const _T& __first_bin_min_val, const _T& __last_bin_max_val)
{
    return internal::__pattern_histogram(
        ::std::forward<Policy>(policy), __first, __last, __histogram_first, num_bins,
        internal::__evenly_divided_binhash<_T>(__first_bin_min_val, __last_bin_max_val, num_bins));
}

template <typename Policy, typename Iter1, typename OutputIter, typename Iter3>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<typename ::std::decay<Policy>::type, OutputIter>
__histogram_impl(Policy&& policy, Iter1 __first, Iter1 __last, OutputIter __histogram_first, Iter3 __boundary_first,
                 Iter3 __boundary_last)
{
    auto keep_boundaries =
        oneapi::dpl::__ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read, Iter3>();
    auto boundary_buf = keep_boundaries(__boundary_first, __boundary_last);

    return internal::__pattern_histogram(
        ::std::forward<Policy>(policy), __first, __last, __histogram_first, (__boundary_last - __boundary_first) - 1,
        internal::__custom_range_binhash{boundary_buf.all_view()}, boundary_buf.all_view());
}

#endif // _ONEDPL_BACKEND_SYCL

} // namespace internal

template <typename _ExecutionPolicy, typename _InputIterator, typename _Size, typename _T, typename _OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _OutputIterator>
histogram(_ExecutionPolicy&& policy, _InputIterator __first, _InputIterator __last, const _Size& num_bins,
          const _T& __first_bin_min_val, const _T& __last_bin_max_val, _OutputIterator __histogram_first)
{
    return internal::__histogram_impl(::std::forward<_ExecutionPolicy>(policy), __first, __last, __histogram_first,
                                      num_bins, __first_bin_min_val, __last_bin_max_val);
}

template <typename _ExecutionPolicy, typename _InputIterator1, typename _InputIterator2, typename _OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _OutputIterator>
histogram(_ExecutionPolicy&& policy, _InputIterator1 __first, _InputIterator1 __last, _InputIterator2 __boundary_first,
          _InputIterator2 __boundary_last, _OutputIterator __histogram_first)
{
    return internal::__histogram_impl(::std::forward<_ExecutionPolicy>(policy), __first, __last, __histogram_first,
                                      __boundary_first, __boundary_last);
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_IMPL_H
