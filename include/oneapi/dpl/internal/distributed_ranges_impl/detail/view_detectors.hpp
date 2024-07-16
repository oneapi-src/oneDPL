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

#pragma once

#include <type_traits>

namespace oneapi::dpl::experimental::dr
{

template <typename T>
struct is_ref_view : std::false_type
{
};
template <rng::range R>
struct is_ref_view<rng::ref_view<R>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_ref_view_v = is_ref_view<T>{};

template <typename T>
struct is_iota_view : std::false_type
{
};
template <std::weakly_incrementable W>
struct is_iota_view<rng::iota_view<W>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_iota_view_v = is_iota_view<T>{};

template <typename T>
struct is_take_view : std::false_type
{
};
template <typename T>
struct is_take_view<rng::take_view<T>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_take_view_v = is_take_view<T>::value;

template <typename T>
struct is_drop_view : std::false_type
{
};
template <typename T>
struct is_drop_view<rng::drop_view<T>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_drop_view_v = is_drop_view<T>::value;

template <typename T>
struct is_subrange_view : std::false_type
{
};
template <typename T>
struct is_subrange_view<rng::subrange<T>> : std::true_type
{
};

template <typename T>
inline constexpr bool is_subrange_view_v = is_subrange_view<T>::value;

template <typename T>
struct is_sliding_view : std::false_type
{
};
#if (defined __cpp_lib_ranges_slide)

template <typename T>
struct is_sliding_view<rng::slide_view<T>> : std::true_type
{
};
template <typename T>
inline constexpr bool is_sliding_view_v = is_sliding_view<std::remove_cvref_t<T>>::value;

#endif

template <typename T>
struct is_zip_view : std::false_type
{
};

#if (defined _cpp_lib_ranges_zip)
template <typename... Views>
struct is_zip_view<rng::zip_view<Views...>> : std::true_type
{
};

#endif
template <typename T>
inline constexpr bool is_zip_view_v = is_zip_view<T>::value;

} // namespace oneapi::dpl::experimental::dr
