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

#include <memory>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/segments_tools.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/detail.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/device_ptr.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/util.hpp>

namespace oneapi::dpl::experimental::dr::shp
{

template <std::contiguous_iterator Iter>
requires(!std::is_const_v<std::iter_value_t<Iter>> && std::is_trivially_copyable_v<std::iter_value_t<Iter>>) sycl::event
    fill_async(Iter first, Iter last, const std::iter_value_t<Iter>& value)
{
    auto&& q = __detail::get_queue_for_pointer(first);
    std::iter_value_t<Iter>* arr = std::to_address(first);
    // not using q.fill because of https://github.com/oneapi-src/distributed-ranges/issues/775
    return dr::__detail::parallel_for(q, sycl::range<>(last - first), [=](auto idx) { arr[idx] = value; });
}

template <std::contiguous_iterator Iter>
requires(!std::is_const_v<std::iter_value_t<Iter>>) void fill(Iter first, Iter last,
                                                              const std::iter_value_t<Iter>& value)
{
    fill_async(first, last, value).wait();
}

template <typename T, typename U>
requires(std::indirectly_writable<device_ptr<T>, U>) sycl::event
    fill_async(device_ptr<T> first, device_ptr<T> last, const U& value)
{
    auto&& q = __detail::get_queue_for_pointer(first);
    auto* arr = first.get_raw_pointer();
    // not using q.fill because of https://github.com/oneapi-src/distributed-ranges/issues/775
    return dr::__detail::parallel_for(q, sycl::range<>(last - first), [=](auto idx) { arr[idx] = value; });
}

template <typename T, typename U>
requires(std::indirectly_writable<device_ptr<T>, U>) void fill(device_ptr<T> first, device_ptr<T> last, const U& value)
{
    fill_async(first, last, value).wait();
}

template <typename T, remote_contiguous_range R>
sycl::event
fill_async(R&& r, const T& value)
{
    auto&& q = __detail::queue(ranges::rank(r));
    auto* arr = std::to_address(stdrng::begin(ranges::local(r)));
    // not using q.fill because of https://github.com/oneapi-src/distributed-ranges/issues/775
    return dr::__detail::parallel_for(q, sycl::range<>(stdrng::distance(r)), [=](auto idx) { arr[idx] = value; });
}

template <typename T, remote_contiguous_range R>
auto
fill(R&& r, const T& value)
{
    fill_async(r, value).wait();
    return stdrng::end(r);
}

template <typename T, distributed_contiguous_range DR>
sycl::event
fill_async(DR&& r, const T& value)
{
    std::vector<sycl::event> events;

    for (auto&& segment : ranges::segments(r))
    {
        auto e = fill_async(segment, value);
        events.push_back(e);
    }

    return __detail::combine_events(events);
}

template <typename T, distributed_contiguous_range DR>
auto
fill(DR&& r, const T& value)
{
    fill_async(r, value).wait();
    return stdrng::end(r);
}

template <typename T, distributed_iterator Iter>
auto
fill(Iter first, Iter last, const T& value)
{
    fill_async(stdrng::subrange(first, last), value).wait();
    return last;
}

} // namespace oneapi::dpl::experimental::dr::shp
