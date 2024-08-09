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

#include <sycl/sycl.hpp>

#include "../../detail/sycl_utils.hpp"
#include "../detail.hpp"
#include "../init.hpp"
#include "../util.hpp"
#include "execution_policy.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

template <typename ExecutionPolicy, distributed_range R, typename Fn>
void
for_each(ExecutionPolicy&& policy, R&& r, Fn fn)
{
    static_assert( // currently only one policy supported
        std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, sycl_device_collection>);

    std::vector<sycl::event> events;
    events.reserve(stdrng::size(ranges::segments(r)));

    for (auto&& segment : ranges::segments(r))
    {
        auto&& q = __detail::queue(ranges::rank(segment));

        assert(stdrng::distance(segment) > 0);

        auto local_segment = ranges::__detail::local_or_identity(segment);

        auto first = stdrng::begin(local_segment);

        auto event = dr::__detail::parallel_for(q, sycl::range<>(stdrng::distance(local_segment)),
                                                [=](auto idx) { fn(first[idx]); });
        events.emplace_back(event);
    }
    __detail::wait(events);
}

template <typename ExecutionPolicy, distributed_iterator Iter, typename Fn>
void
for_each(ExecutionPolicy&& policy, Iter begin, Iter end, Fn fn)
{
    for_each(std::forward<ExecutionPolicy>(policy), stdrng::subrange(begin, end), fn);
}

template <distributed_range R, typename Fn>
void
for_each(R&& r, Fn fn)
{
    for_each(par_unseq, std::forward<R>(r), fn);
}

template <distributed_iterator Iter, typename Fn>
void
for_each(Iter begin, Iter end, Fn fn)
{
    for_each(par_unseq, begin, end, fn);
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter, std::integral I, typename Fn>
Iter
for_each_n(ExecutionPolicy&& policy, Iter begin, I n, Fn fn)
{
    auto end = begin;
    stdrng::advance(end, n);
    for_each(std::forward<ExecutionPolicy>(policy), begin, end, std::forward<Fn>(fn));
    return end;
}

template <dr::distributed_iterator Iter, std::integral I, typename Fn>
Iter
for_each_n(Iter&& r, I n, Fn fn)
{
    return for_each_n(dr::sp::par_unseq, std::forward<Iter>(r), n, fn);
}

} // namespace oneapi::dpl::experimental::dr::sp
