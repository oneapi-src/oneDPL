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

#ifndef _ONEDPL_DR_CONCEPTS_CONCEPTS_HPP
#define _ONEDPL_DR_CONCEPTS_CONCEPTS_HPP

#include "detail/ranges.hpp"

namespace oneapi::dpl::experimental::dr
{

template <typename I>
concept remote_iterator = std::forward_iterator<I> && requires(I& iter)
{
    ranges::rank(iter);
};

template <typename R>
concept remote_range = stdrng::forward_range<R> && requires(R& r)
{
    ranges::rank(r);
};

template <typename R>
concept distributed_range = stdrng::forward_range<R> && requires(R& r)
{
    ranges::segments(r);
};

template <typename I>
concept remote_contiguous_iterator = std::random_access_iterator<I> && requires(I& iter)
{
    ranges::rank(iter);
    {
        ranges::local(iter)
        } -> std::contiguous_iterator;
};

template <typename I>
concept distributed_iterator = std::forward_iterator<I> && requires(I& iter)
{
    ranges::segments(iter);
};

template <typename R>
concept remote_contiguous_range = remote_range<R> && stdrng::random_access_range<R> && requires(R& r)
{
    {
        ranges::local(r)
        } -> stdrng::contiguous_range;
};

template <typename R>
concept distributed_contiguous_range = distributed_range<R> && stdrng::random_access_range<R> && requires(R& r)
{
    {
        ranges::segments(r)
        } -> stdrng::random_access_range;
} && remote_contiguous_range<stdrng::range_value_t<decltype(ranges::segments(std::declval<R>()))>>;

template <typename Iter>
concept distributed_contiguous_iterator = distributed_iterator<Iter> && std::random_access_iterator<Iter> &&
    requires(Iter& iter)
{
    {
        ranges::segments(iter)
        } -> stdrng::random_access_range;
} && remote_contiguous_range<stdrng::range_value_t<decltype(ranges::segments(std::declval<Iter>()))>>;

} // namespace oneapi::dpl::experimental::dr

#endif /* _ONEDPL_DR_CONCEPTS_CONCEPTS_HPP */
