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
// Public header file provides stuff functionality for counter-based RNG engines.
//
//   integral_innput-range - concepts
//   uint_fast<w> - the fast unsigned type with at least w bits
//   uint_least<w> - the smallest unsigned type with at least w bits
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

// uint_fast<w> :  the fast unsigned integer type with at least w bits.
// uint_least<w> : the smallest unsigned integer type with at least w bits.
// N.B.  This is a whole lot simpler than it used to be!
constexpr unsigned next_stdint(unsigned w) {
    if (w <= 8)
        return 8;
    else if (w <= 16)
        return 16;
    else if (w <= 32)
        return 32;
    else if (w <= 64)
        return 64;
    else if (w <= 128)
        return 128; // DANGER - assuming __uint128_t
    return 0;
}

template <unsigned W>
struct ui {};
template <>
struct ui<8> {
    using least = uint_least8_t;
    using fast = uint_fast8_t;
};
template <>
struct ui<16> {
    using least = uint_least16_t;
    using fast = uint_fast16_t;
};
template <>
struct ui<32> {
    using least = uint_least32_t;
    using fast = uint_fast32_t;
};
template <>
struct ui<64> {
    using least = uint_least64_t;
    using fast = uint_fast64_t;
};
template <>
struct ui<128> {
    using least = __uint128_t;
    using fast = __uint128_t;
}; // DANGER - __uint128_t

template <unsigned w>
using uint_least = typename ui<next_stdint(w)>::least;
template <unsigned w>
using uint_fast = typename ui<next_stdint(w)>::fast;

// Implement w-bit mulhilo with an 2w-wide integer.  If we don't
// have a 2w-wide integer, we're out of luck.
template <unsigned W, std::unsigned_integral U>
static std::pair<U, U> mulhilo(U a, U b) {
    using uwide = uint_fast<2 * W>;
    const size_t xwidth = std::numeric_limits<uwide>::digits;
    uwide ab = uwide(a) * uwide(b);
    return { U(ab >> W), U((ab << (xwidth - W)) >> (xwidth - W)) };
}

template <std::unsigned_integral U, unsigned W>
requires(W <= std::numeric_limits<U>::digits) constexpr U fffmask =
    W ? (U(~(U(0))) >> (std::numeric_limits<U>::digits - W)) : 0;

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