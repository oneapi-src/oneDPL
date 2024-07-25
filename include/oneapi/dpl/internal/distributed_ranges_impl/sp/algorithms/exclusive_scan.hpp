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

#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include "../../concepts/concepts.hpp"
#include "../../detail/onedpl_direct_iterator.hpp"
#include "../allocators.hpp"
#include "../detail.hpp"
#include "../init.hpp"
#include "../vector.hpp"
#include "../views/views.hpp"
#include "execution_policy.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

namespace __detail
{
template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename U,
          typename BinaryOp>
void
exclusive_scan_impl_(ExecutionPolicy&& policy, R&& r, O&& o, U init, BinaryOp binary_op)
{
    using T = stdrng::range_value_t<O>;

    static_assert(std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, distributed_device_policy>);

    auto zipped_view = views::zip(r, o);
    auto zipped_segments = zipped_view.zipped_segments();

    if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, distributed_device_policy>)
    {

        U* d_inits = sycl::malloc_device<U>(stdrng::size(zipped_segments), devices()[0], context());

        std::vector<sycl::event> events;

        std::size_t segment_id = 0;
        for (auto&& segs : zipped_segments)
        {
            auto&& [in_segment, out_segment] = segs;

            auto last_element = stdrng::prev(stdrng::end(__detail::local(in_segment)));
            auto dest = d_inits + segment_id;

            auto&& q = __detail::queue(ranges::rank(in_segment));

            auto e = q.single_task([=] { *dest = *last_element; });
            events.push_back(e);
            segment_id++;
        }

        __detail::wait(events);
        events.clear();

        std::vector<U> inits(stdrng::size(zipped_segments));

        copy(d_inits, d_inits + inits.size(), inits.data() + 1);

        sycl::free(d_inits, context());

        // when empty range fails due to: https://github.com/oneapi-src/distributed-ranges/issues/791
        inits[0] = init;

        auto root = devices()[0];
        device_allocator<T> allocator(context(), root);
        vector<T, device_allocator<T>> partial_sums(std::size_t(zipped_segments.size()), allocator);

        // DPL futures must be kept alive, since in the future their destruction
        // may trigger a synchronization.
        using dpl_future_type = decltype(exclusive_scan_async(
            __detail::dpl_policy(0),
            dr::__detail::direct_iterator(stdrng::begin(std::get<0>(*zipped_segments.begin()))),
            dr::__detail::direct_iterator(stdrng::begin(std::get<0>(*zipped_segments.begin()))),
            dr::__detail::direct_iterator(stdrng::begin(std::get<1>(*zipped_segments.begin()))), init, binary_op));

        std::vector<dpl_future_type> futures;

        segment_id = 0;
        for (auto&& segs : zipped_segments)
        {
            auto&& [in_segment, out_segment] = segs;

            auto&& q = __detail::queue(ranges::rank(in_segment));
            auto&& local_policy = __detail::dpl_policy(ranges::rank(in_segment));

            auto dist = stdrng::distance(in_segment);
            assert(dist > 0);

            auto first = stdrng::begin(in_segment);
            auto last = stdrng::end(in_segment);
            auto d_first = stdrng::begin(out_segment);

            auto init = inits[segment_id];

            futures.push_back(exclusive_scan_async(local_policy, dr::__detail::direct_iterator(first),
                                                   dr::__detail::direct_iterator(last),
                                                   dr::__detail::direct_iterator(d_first), init, binary_op));

            auto dst_iter = ranges::local(partial_sums).data() + segment_id;

            auto src_iter = ranges::local(out_segment).data();
            stdrng::advance(src_iter, dist - 1);

            auto e = q.submit([&](auto&& h) {
                h.depends_on(futures.back().event());
                h.single_task([=]() {
                    stdrng::range_value_t<O> value = *src_iter;
                    *dst_iter = value;
                });
            });

            events.push_back(e);

            segment_id++;
        }

        __detail::wait(events);
        events.clear();
        futures.clear();

        auto&& local_policy = __detail::dpl_policy(0);

        auto first = ranges::local(partial_sums).data();
        auto last = first + partial_sums.size();

        inclusive_scan_async(local_policy, first, last, first, binary_op).wait();

        std::size_t idx = 0;
        for (auto&& segs : zipped_segments)
        {
            auto&& [in_segment, out_segment] = segs;

            if (idx > 0)
            {
                auto&& q = __detail::queue(ranges::rank(out_segment));

                auto first = stdrng::begin(out_segment);
                dr::__detail::direct_iterator d_first(first);

                auto d_sum = ranges::__detail::local(partial_sums).begin() + idx - 1;

                sycl::event e =
                    dr::__detail::parallel_for(q, sycl::range<>(stdrng::distance(out_segment)),
                                               [=](auto idx) { d_first[idx] = binary_op(d_first[idx], *d_sum); });

                events.push_back(e);
            }
            idx++;
        }

        __detail::wait(events);
    }
    else
    {
        assert(false);
    }
}

} // namespace __detail
// Ranges versions

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename T,
          typename BinaryOp>
void
exclusive_scan(ExecutionPolicy&& policy, R&& r, O&& o, T init, BinaryOp binary_op)
{
    __detail::exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), std::forward<O>(o), init,
                                   binary_op);
}

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename T>
void
exclusive_scan(ExecutionPolicy&& policy, R&& r, O&& o, T init)
{
    __detail::exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), std::forward<O>(o), init,
                                   std::plus<>{});
}

template <distributed_contiguous_range R, distributed_contiguous_range O, typename T, typename BinaryOp>
void
exclusive_scan(R&& r, O&& o, T init, BinaryOp binary_op)
{
    __detail::exclusive_scan_impl_(par_unseq, std::forward<R>(r), std::forward<O>(o), init, binary_op);
}

template <distributed_contiguous_range R, distributed_contiguous_range O, typename T>
void
exclusive_scan(R&& r, O&& o, T init)
{
    __detail::exclusive_scan_impl_(par_unseq, std::forward<R>(r), std::forward<O>(o), init, std::plus<>{});
}

// Iterator versions

template <typename ExecutionPolicy, distributed_iterator Iter, distributed_iterator OutputIter, typename T,
          typename BinaryOp>
void
exclusive_scan(ExecutionPolicy&& policy, Iter first, Iter last, OutputIter d_first, T init, BinaryOp binary_op)
{
    auto dist = stdrng::distance(first, last);
    auto d_last = d_first;
    stdrng::advance(d_last, dist);
    __detail::exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last),
                                   stdrng::subrange(d_first, d_last), init, binary_op);
}

template <typename ExecutionPolicy, distributed_iterator Iter, distributed_iterator OutputIter, typename T>
void
exclusive_scan(ExecutionPolicy&& policy, Iter first, Iter last, OutputIter d_first, T init)
{
    exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, d_first, init, std::plus<>{});
}

template <distributed_iterator Iter, distributed_iterator OutputIter, typename T, typename BinaryOp>
void
exclusive_scan(Iter first, Iter last, OutputIter d_first, T init, BinaryOp binary_op)
{
    exclusive_scan(par_unseq, first, last, d_first, init, binary_op);
}

template <distributed_iterator Iter, distributed_iterator OutputIter, typename T>
void
exclusive_scan(Iter first, Iter last, OutputIter d_first, T init)
{
    exclusive_scan(par_unseq, first, last, d_first, init);
}

} // namespace oneapi::dpl::experimental::dr::sp
