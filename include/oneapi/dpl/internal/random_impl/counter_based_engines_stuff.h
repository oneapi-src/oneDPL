// -*- C++ -*-
//===-- philox-engine.h ------------------------------------------===//
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
//
//   fffmask<Uint, w> - the Uint with the low w bits set
//   mulhilo<w, Uint> -> pair<U, U> - returns the w hi
//       and w low bits of the 2w-bit product of a and b.

#ifndef _ONEDPL_CTR_ENGINES_STUFF_H
#define _ONEDPL_CTR_ENGINES_STUFF_H

#include <concepts>
#include <iterator>
#include <utility>
#include <tuple>
#include <limits>

namespace detail {



// template <unsigned W, typename U> 
// static std::pair<U, U> mulhilo(U a, U b) {
//     static_assert(::std::is_unsigned_v<U>);

//     if constexpr (std::is_same_v<U, uint_fast32_t>) {
//         using uwide = __uint64_t;
//         const size_t xwidth = std::numeric_limits<uwide>::digits;
//         uwide ab = uwide(a) * uwide(b);
//         return { U(ab >> W), U((ab << (xwidth - W)) >> (xwidth - W)) };
//     }
//     else {
//         U lo = a*b;

//         __uint32_t tmp = ((__uint32_t)a * (__uint32_t)(b>>32) +(__uint32_t)b * (__uint32_t)(a>>32)) >> 32;
//         U hi = (__uint32_t)(b>>32)*(__uint32_t)(a>>32)+tmp;

//         return { lo, hi };
//     }
// }

template <typename U, unsigned W,
          typename = ::std::enable_if_t<std::is_unsigned_v<U> && (W <= std::numeric_limits<U>::digits)>>
constexpr U fffmask = W ? (U(~(U(0))) >> (std::numeric_limits<U>::digits - W)) : 0;

// For unpacking variadic of constants into two arrays:
template <typename UIntType, typename Tuple, size_t... Is>
constexpr auto get_even_array_from_tuple(Tuple t, std::index_sequence<Is...>) {
    return std::array<UIntType, std::index_sequence<Is...>::size()>{ std::get<Is * 2>(t)... };
}

template <typename UIntType, typename Tuple, size_t... Is>
constexpr auto get_odd_array_from_tuple(Tuple t, std::index_sequence<Is...>) {
    return std::array<UIntType, std::index_sequence<Is...>::size()>{ std::get<Is * 2 + 1>(t)... };
}

} // namespace detail

#endif // _ONEDPL_CTR_ENGINES_STUFF_H