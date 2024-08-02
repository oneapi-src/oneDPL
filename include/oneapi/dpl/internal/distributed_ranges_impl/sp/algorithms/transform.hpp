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

#ifndef _ONEDPL_DR_SP_TRANSFORM_HPP
#define _ONEDPL_DR_SP_TRANSFORM_HPP

#include "../detail.hpp"
#include "../init.hpp"
#include "../util.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

/**
 * Applies the given function to a range and stores the result in another range,
 * beginning at out.
 * \param policy use `par_unseq` here only
 * \param in the range of elements to transform
 * \param out the beginning of the destination range, may be equal to the
 * beginning of `in` range \param fn operation to apply to input elements
 * \return an
 * [unary_transform_result](https://en.cppreference.com/w/cpp/algorithm/ranges/transform)
 * containing an input iterator equal to the end of `in` range and an output
 * iterator to the element past the last element transformed
 */

template <class ExecutionPolicy>
auto
transform(ExecutionPolicy&& policy, distributed_range auto&& in, distributed_iterator auto out, auto&& fn)
{

    static_assert( // currently only one policy supported
        std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, sycl_device_collection>);

    std::vector<sycl::event> events;
    using OutT = typename decltype(out)::value_type;
    std::vector<void*> buffers;
    const auto out_end = out + stdrng::size(in);

    for (auto&& [in_seg, out_seg] : views::zip(in, stdrng::subrange(out, out_end)).zipped_segments())
    {
        auto in_device = policy.get_devices()[in_seg.rank()];
        auto&& q = __detail::queue(ranges::rank(in_seg));
        const std::size_t seg_size = stdrng::size(in_seg);
        assert(seg_size == stdrng::size(out_seg));
        auto local_in_seg = ranges::local_or_identity(in_seg);

        if (in_seg.rank() == out_seg.rank())
        {
            auto local_out_seg = ranges::local_or_identity(out_seg);
            events.emplace_back(
                q.parallel_for(seg_size, [=](auto idx) { local_out_seg[idx] = fn(local_in_seg[idx]); }));
        }
        else
        {
            OutT* buffer = sycl::malloc_device<OutT>(seg_size, in_device, context());
            buffers.push_back(buffer);

            sycl::event compute_event =
                q.parallel_for(seg_size, [=](auto idx) { buffer[idx] = fn(local_in_seg[idx]); });
            events.emplace_back(q.copy(buffer, ranges::local_or_identity(out_seg.begin()), seg_size, compute_event));
        }
    }
    __detail::wait(events);

    for (auto* b : buffers)
        sycl::free(b, context());

    return stdrng::unary_transform_result<decltype(stdrng::end(in)), decltype(out_end)>{stdrng::end(in), out_end};
}

template <distributed_range R, distributed_iterator Iter, typename Fn>
auto
transform(R&& in, Iter out, Fn&& fn)
{
    return transform(par_unseq, std::forward<R>(in), std::forward<Iter>(out), std::forward<Fn>(fn));
}

template <typename ExecutionPolicy, distributed_iterator Iter1, distributed_iterator Iter2, typename Fn>
auto
transform(ExecutionPolicy&& policy, Iter1 in_begin, Iter1 in_end, Iter2 out_end, Fn&& fn)
{
    return transform(std::forward<ExecutionPolicy>(policy),
                     stdrng::subrange(std::forward<Iter1>(in_begin), std::forward<Iter1>(in_end)),
                     std::forward<Iter2>(out_end), std::forward<Fn>(fn));
}

template <distributed_iterator Iter1, distributed_iterator Iter2, typename Fn>
auto
transform(Iter1 in_begin, Iter1 in_end, Iter2 out_end, Fn&& fn)
{
    return transform(par_unseq, std::forward<Iter1>(in_begin), std::forward<Iter1>(in_end),
                     std::forward<Iter2>(out_end), std::forward<Fn>(fn));
}

} // namespace oneapi::dpl::experimental::dr::sp

#endif /* _ONEDPL_DR_SP_TRANSFORM_HPP */
