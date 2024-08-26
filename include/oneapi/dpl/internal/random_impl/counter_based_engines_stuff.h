// -*- C++ -*-
//===-- counter_based_engines_stuff.h -------------------------------------===//
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
  
/* word_mask<U, W> - an unsigned integral type with the low W bits set */
template <typename U, unsigned W,
          typename = ::std::enable_if_t<::std::is_unsigned_v<U> && (W <= ::std::numeric_limits<U>::digits)>>
constexpr U word_mask = W ? (U(~(U(0))) >> (::std::numeric_limits<U>::digits - W)) : 0;

/* For unpacking variadic of constants into two arrays */
template <typename UIntType, typename Tuple, ::std::size_t... Is>
constexpr auto get_even_array_from_tuple(Tuple t, ::std::index_sequence<Is...>) {
    return ::std::array<UIntType, ::std::index_sequence<Is...>::size()>{ ::std::get<Is * 2>(t)... };
}
template <typename UIntType, typename Tuple, ::std::size_t... Is>
constexpr auto get_odd_array_from_tuple(Tuple t, std::index_sequence<Is...>) {
    return ::std::array<UIntType, ::std::index_sequence<Is...>::size()>{ ::std::get<Is * 2 + 1>(t)... };
}

/* Implement W-bit mulhilo - returns the W hi and W low 
   bits of the 2W-bit product of a and b */
template <typename UIntType, unsigned W>
static ::std::pair<UIntType, UIntType> mulhilo(UIntType a, UIntType b)
{
    static_assert(W <= 64, "W must be 0 < W <= 64");

    using result_type = UIntType;
    result_type res_hi, res_lo;

    /* multiplication fits standard types */
    if constexpr (W <= 32) {
        uint_fast64_t mult_result = (uint_fast64_t)a * (uint_fast64_t)b;
        res_hi = mult_result >> W;
        res_lo =  mult_result & internal::word_mask<result_type, W>;
    }
    /* pen-pencil multiplication by 32-bit chunks */
    else if constexpr(W > 32) {
        res_lo = a * b;

        result_type x0 = a & internal::word_mask<result_type, 32>;
        result_type x1 = a >> 32;
        result_type y0 = b & internal::word_mask<result_type, 32>;
        result_type y1 = b >> 32;

        result_type p11 = x1 * y1;
        result_type p01 = x0 * y1;
        result_type p10 = x1 * y0;
        result_type p00 = x0 * y0;

        // 64-bit product + two 32-bit values
        result_type middle = p10 + (p00 >> 32) + (p01 & internal::word_mask<result_type, 32>);

        // 64-bit product + two 32-bit values
        res_hi = p11 + (middle >> 32) + (p01 >> 32);
    }
    
    return { res_hi, res_lo };
}

} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_CTR_ENGINES_STUFF_H
