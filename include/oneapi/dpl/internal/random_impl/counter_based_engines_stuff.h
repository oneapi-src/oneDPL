// -*- C++ -*-
//===-- counter_based_engines_stuff.h -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Header file provides stuff functionality for counter-based RNG engines.

#ifndef _ONEDPL_CTR_ENGINES_STUFF_H
#define _ONEDPL_CTR_ENGINES_STUFF_H

#include <concepts>
#include <iterator>
#include <utility>
#include <tuple>
#include <limits>

namespace oneapi
{
namespace dpl
{
namespace internal
{
namespace experimental
{

/* word_mask<_U, __W> - an unsigned integral type with the low __W bits set */
template <typename _U, unsigned __W,
          typename = ::std::enable_if_t<::std::is_unsigned_v<_U> && (__W <= ::std::numeric_limits<_U>::digits)>>
constexpr _U word_mask = __W ? (_U(~(_U(0))) >> (::std::numeric_limits<_U>::digits - __W)) : 0;

/* For unpacking variadic of constants into two arrays */
template <typename _UIntType, typename _Tuple, ::std::size_t... __Is>
constexpr auto
get_even_array_from_tuple(_Tuple __t, ::std::index_sequence<__Is...>)
{
    return ::std::array<_UIntType, ::std::index_sequence<__Is...>::size()>{::std::get<__Is * 2>(__t)...};
}
template <typename _UIntType, typename _Tuple, ::std::size_t... __Is>
constexpr auto
get_odd_array_from_tuple(_Tuple __t, std::index_sequence<__Is...>)
{
    return ::std::array<_UIntType, ::std::index_sequence<__Is...>::size()>{::std::get<__Is * 2 + 1>(__t)...};
}

/* Implement __W-bit mulhilo - returns the __W hi and __W low 
   bits of the 2__W-bit product of __a and __b */
template <typename _UIntType, unsigned __W>
static ::std::pair<_UIntType, _UIntType>
mulhilo(_UIntType __a, _UIntType __b)
{
    static_assert(__W <= 64, "__W must be 0 < __W <= 64");

    using result_type = _UIntType;
    result_type __res_hi, __res_lo;

    /* multiplication fits standard types */
    if constexpr (__W <= 32)
    {
        uint_fast64_t __mult_result = (uint_fast64_t)__a * (uint_fast64_t)__b;
        __res_hi = __mult_result >> __W;
        __res_lo = __mult_result & internal::experimental::word_mask<result_type, __W>;
    }
    /* pen-pencil multiplication by 32-bit chunks */
    else if constexpr (__W > 32)
    {
        __res_lo = __a * __b;

        result_type __x0 = __a & internal::experimental::word_mask<result_type, 32>;
        result_type __x1 = __a >> 32;
        result_type __y0 = __b & internal::experimental::word_mask<result_type, 32>;
        result_type __y1 = __b >> 32;

        result_type __p11 = __x1 * __y1;
        result_type __p01 = __x0 * __y1;
        result_type __p10 = __x1 * __y0;
        result_type __p00 = __x0 * __y0;

        // 64-bit product + two 32-bit values
        result_type __middle = __p10 + (__p00 >> 32) + (__p01 & internal::experimental::word_mask<result_type, 32>);

        // 64-bit product + two 32-bit values
        __res_hi = __p11 + (__middle >> 32) + (__p01 >> 32);
    }

    return {__res_hi, __res_lo};
}

} // namespace experimental
} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_CTR_ENGINES_STUFF_H
