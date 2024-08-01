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

#ifndef _ONEDPL_DR_SP_REDUCE_HPP
#define _ONEDPL_DR_SP_REDUCE_HPP

#include <sycl/sycl.hpp>

#include "oneapi/dpl/async"
#include "oneapi/dpl/execution"
#include "oneapi/dpl/numeric"

#include "../../concepts/concepts.hpp"
#include "../../detail/onedpl_direct_iterator.hpp"
#include "../init.hpp"
#include "execution_policy.hpp"

namespace
{

// Precondition: stdrng::distance(first, last) >= 2
// Postcondition: return future to [first, last) reduced with fn
template <typename T, typename ExecutionPolicy, std::bidirectional_iterator Iter, typename Fn>
auto
reduce_no_init_async(ExecutionPolicy&& policy, Iter first, Iter last, Fn&& fn)
{
    Iter new_last = last;
    --new_last;

    std::iter_value_t<Iter> init = *new_last;

    oneapi::dpl::experimental::dr::__detail::direct_iterator d_first(first);
    oneapi::dpl::experimental::dr::__detail::direct_iterator d_last(new_last);

    return oneapi::dpl::experimental::reduce_async(std::forward<ExecutionPolicy>(policy), d_first, d_last,
                                                   static_cast<T>(init), std::forward<Fn>(fn));
}

template <typename T, typename ExecutionPolicy, std::bidirectional_iterator Iter, typename Fn>
requires(sycl::has_known_identity_v<Fn, T>) auto reduce_no_init_async(ExecutionPolicy&& policy, Iter first, Iter last,
                                                                      Fn&& fn)
{
    oneapi::dpl::experimental::dr::__detail::direct_iterator d_first(first);
    oneapi::dpl::experimental::dr::__detail::direct_iterator d_last(last);

    return oneapi::dpl::experimental::reduce_async(std::forward<ExecutionPolicy>(policy), d_first, d_last,
                                                   sycl::known_identity_v<Fn, T>, std::forward<Fn>(fn));
}

} // namespace

namespace oneapi::dpl::experimental::dr::sp
{

template <typename ExecutionPolicy, distributed_range R, typename T, typename BinaryOp>
T
reduce(ExecutionPolicy&& policy, R&& r, T init, BinaryOp binary_op)
{

    static_assert(std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, sycl_device_collection>);

    using future_t = decltype(reduce_async(__detail::dpl_policy(0), ranges::segments(r)[0].begin(),
                                           ranges::segments(r)[0].end(), init, binary_op));
    std::vector<future_t> futures;

    for (auto&& segment : ranges::segments(r))
    {
        auto&& local_policy = __detail::dpl_policy(ranges::rank(segment));

        auto dist = stdrng::distance(segment);
        if (dist <= 0)
        {
            continue;
        }
        else if (dist == 1)
        {
            init = binary_op(init, *stdrng::begin(segment));
            continue;
        }

        auto future = reduce_no_init_async<T>(local_policy, stdrng::begin(segment), stdrng::end(segment), binary_op);

        futures.push_back(std::move(future));
    }

    for (auto&& f : futures)
    {
        init = binary_op(init, f.get());
    }

    return init;
}

template <typename ExecutionPolicy, distributed_range R, typename T>
T
reduce(ExecutionPolicy&& policy, R&& r, T init)
{
    return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), init, std::plus<>());
}

template <typename ExecutionPolicy, distributed_range R>
stdrng::range_value_t<R>
reduce(ExecutionPolicy&& policy, R&& r)
{
    return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), stdrng::range_value_t<R>{}, std::plus<>());
}

// Iterator versions

template <typename ExecutionPolicy, distributed_iterator Iter>
std::iter_value_t<Iter>
reduce(ExecutionPolicy&& policy, Iter first, Iter last)
{
    return reduce(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last), std::iter_value_t<Iter>{},
                  std::plus<>());
}

template <typename ExecutionPolicy, distributed_iterator Iter, typename T>
T
reduce(ExecutionPolicy&& policy, Iter first, Iter last, T init)
{
    return reduce(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last), init, std::plus<>());
}

template <typename ExecutionPolicy, distributed_iterator Iter, typename T, typename BinaryOp>
T
reduce(ExecutionPolicy&& policy, Iter first, Iter last, T init, BinaryOp binary_op)
{
    return reduce(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last), init, binary_op);
}

// Execution policy-less algorithms

template <distributed_range R>
stdrng::range_value_t<R>
reduce(R&& r)
{
    return reduce(par_unseq, std::forward<R>(r));
}

template <distributed_range R, typename T>
T
reduce(R&& r, T init)
{
    return reduce(par_unseq, std::forward<R>(r), init);
}

template <distributed_range R, typename T, typename BinaryOp>
T
reduce(R&& r, T init, BinaryOp binary_op)
{
    return reduce(par_unseq, std::forward<R>(r), init, binary_op);
}

template <distributed_iterator Iter>
std::iter_value_t<Iter>
reduce(Iter first, Iter last)
{
    return reduce(par_unseq, first, last);
}

template <distributed_iterator Iter, typename T>
T
reduce(Iter first, Iter last, T init)
{
    return reduce(par_unseq, first, last, init);
}

template <distributed_iterator Iter, typename T, typename BinaryOp>
T
reduce(Iter first, Iter last, T init, BinaryOp binary_op)
{
    return reduce(par_unseq, first, last, init, binary_op);
}

} // namespace oneapi::dpl::experimental::dr::sp

#endif
