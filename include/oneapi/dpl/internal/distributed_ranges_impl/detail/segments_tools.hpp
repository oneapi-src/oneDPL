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

#ifndef _ONEDPL_DR_DETAIL_SEGMENT_TOOLS_HPP
#define _ONEDPL_DR_DETAIL_SEGMENT_TOOLS_HPP

#include "enumerate.hpp"
#include "std_ranges_shim.hpp"
#include "remote_subrange.hpp"
#include "view_detectors.hpp"

namespace oneapi::dpl::experimental::dr
{

namespace __detail
{

// Take all elements up to and including segment `segment_id` at index
// `local_id`
template <typename R>
auto
take_segments(R&& segments, std::size_t last_seg, std::size_t local_id)
{
    auto remainder = local_id;

    auto take_partial = [=](auto&& v) {
        auto&& [i, segment] = v;
        if (i == last_seg)
        {
            auto first = stdrng::begin(segment);
            auto last = stdrng::begin(segment);
            stdrng::advance(last, remainder);
            return remote_subrange(first, last, ranges::rank(segment));
        }
        else
        {
            return remote_subrange(segment);
        }
    };

    return enumerate(segments) | stdrng::views::take(last_seg + 1) | stdrng::views::transform(std::move(take_partial));
}

// Take the first n elements
template <typename R>
auto
take_segments(R&& segments, std::size_t n)
{
    std::size_t last_seg = 0;
    std::size_t remainder = n;

    for (auto&& seg : segments)
    {
        if (seg.size() >= remainder)
        {
            break;
        }
        remainder -= seg.size();
        last_seg++;
    }

    return take_segments(std::forward<R>(segments), last_seg, remainder);
}

// Drop all elements up to segment `segment_id` and index `local_id`
template <typename R>
auto
drop_segments(R&& segments, std::size_t first_seg, std::size_t local_id)
{
    auto remainder = local_id;

    auto drop_partial = [=](auto&& v) {
        auto&& [i, segment] = v;
        if (i == first_seg)
        {
            auto first = stdrng::begin(segment);
            stdrng::advance(first, remainder);
            auto last = stdrng::end(segment);
            return remote_subrange(first, last, ranges::rank(segment));
        }
        else
        {
            return remote_subrange(segment);
        }
    };

    return enumerate(segments) | stdrng::views::drop(first_seg) | stdrng::views::transform(std::move(drop_partial));
}

// Drop the first n elements
template <typename R>
auto
drop_segments(R&& segments, std::size_t n)
{
    std::size_t first_seg = 0;
    std::size_t remainder = n;

    for (auto&& seg : segments)
    {
        if (seg.size() > remainder)
        {
            break;
        }
        remainder -= seg.size();
        first_seg++;
    }

    return drop_segments(std::forward<R>(segments), first_seg, remainder);
}
} // namespace __detail

} // namespace oneapi::dpl::experimental::dr

namespace __ONEDPL_DR_STD_RANGES_NAMESPACE
{

// A standard library range adaptor does not change the rank of a
// remote range, so we can simply return the rank of the base view.
template <stdrng::range V>
requires(oneapi::dpl::experimental::dr::remote_range<decltype(std::declval<V>().base())>) auto rank_(V&& v)
{
    return oneapi::dpl::experimental::dr::ranges::rank(std::forward<V>(v).base());
}

template <stdrng::range V>
requires(oneapi::dpl::experimental::dr::is_ref_view_v<std::remove_cvref_t<V>>&&
             oneapi::dpl::experimental::dr::distributed_range<decltype(std::declval<V>().base())>) auto segments_(V&& v)
{
    return oneapi::dpl::experimental::dr::ranges::segments(v.base());
}

template <stdrng::range V>
requires(oneapi::dpl::experimental::dr::is_take_view_v<std::remove_cvref_t<V>>&&
             oneapi::dpl::experimental::dr::distributed_range<decltype(std::declval<V>().base())>) auto segments_(V&& v)
{
    return oneapi::dpl::experimental::dr::__detail::take_segments(
        oneapi::dpl::experimental::dr::ranges::segments(v.base()), v.size());
}

template <stdrng::range V>
requires(oneapi::dpl::experimental::dr::is_drop_view_v<std::remove_cvref_t<V>>&&
             oneapi::dpl::experimental::dr::distributed_range<decltype(std::declval<V>().base())>) auto segments_(V&& v)
{
    return oneapi::dpl::experimental::dr::__detail::drop_segments(
        oneapi::dpl::experimental::dr::ranges::segments(v.base()), v.base().size() - v.size());
}

template <stdrng::range V>
requires(oneapi::dpl::experimental::dr::is_subrange_view_v<std::remove_cvref_t<V>>&&
             oneapi::dpl::experimental::dr::distributed_iterator<stdrng::iterator_t<V>>) auto segments_(V&& v)
{
    auto first = stdrng::begin(v);
    auto last = stdrng::end(v);
    auto size = stdrng::distance(first, last);

    return oneapi::dpl::experimental::dr::__detail::take_segments(
        oneapi::dpl::experimental::dr::ranges::segments(first), size);
}

} // namespace DR_RANGES_NAMESPACE

#endif /* _ONEDPL_DR_DETAIL_SEGMENT_TOOLS_HPP */
