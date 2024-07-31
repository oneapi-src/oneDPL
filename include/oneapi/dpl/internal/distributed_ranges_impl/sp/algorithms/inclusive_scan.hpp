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

#include <optional>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/onedpl_direct_iterator.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/algorithms/execution_policy.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/allocators.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/detail.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/init.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/vector.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/sp/views/views.hpp>

namespace oneapi::dpl::experimental::dr::sp
{

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename BinaryOp,
          typename U = stdrng::range_value_t<R>>
void
inclusive_scan_impl_(ExecutionPolicy&& policy, R&& r, O&& o, BinaryOp binary_op, std::optional<U> init = {})
{
    using T = stdrng::range_value_t<O>;

    static_assert(std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, sycl_device_collection>);

    auto zipped_view = views::zip(r, o);
    auto zipped_segments = zipped_view.zipped_segments();

    if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, sycl_device_collection>)
    {

        std::vector<sycl::event> events;

        auto root = devices()[0];
        device_allocator<T> allocator(context(), root);
        vector<T, device_allocator<T>> partial_sums(std::size_t(zipped_segments.size()), allocator);

        // DPL futures must be kept alive, since in the future their destruction
        // may trigger a synchronization.
        using dpl_future_type = decltype(inclusive_scan_async(
            __detail::dpl_policy(0),
            dr::__detail::direct_iterator(stdrng::begin(std::get<0>(*zipped_segments.begin()))),
            dr::__detail::direct_iterator(stdrng::begin(std::get<0>(*zipped_segments.begin()))),
            dr::__detail::direct_iterator(stdrng::begin(std::get<1>(*zipped_segments.begin()))), binary_op,
            init.value()));

        std::vector<dpl_future_type> futures;

        std::size_t segment_id = 0;
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

            if (segment_id == 0 && init.has_value())
            {
                futures.push_back(inclusive_scan_async(
                    local_policy, dr::__detail::direct_iterator(first), dr::__detail::direct_iterator(last),
                    dr::__detail::direct_iterator(d_first), binary_op, init.value()));
            }
            else
            {
                futures.push_back(inclusive_scan_async(local_policy, dr::__detail::direct_iterator(first),
                                                       dr::__detail::direct_iterator(last),
                                                       dr::__detail::direct_iterator(d_first), binary_op));
            }

            auto dst_iter = ranges::local(partial_sums).data() + segment_id;

            auto src_iter = ranges::local(out_segment).data();
            stdrng::advance(src_iter, dist - 1);

            sycl::event event = futures.back();

            auto e = q.submit([&](auto&& h) {
                h.depends_on(event);
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

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename BinaryOp,
          typename T>
void
inclusive_scan(ExecutionPolicy&& policy, R&& r, O&& o, BinaryOp binary_op, T init)
{
    inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), std::forward<O>(o), binary_op,
                         std::optional(init));
}

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O, typename BinaryOp>
void
inclusive_scan(ExecutionPolicy&& policy, R&& r, O&& o, BinaryOp binary_op)
{
    inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), std::forward<O>(o), binary_op);
}

template <typename ExecutionPolicy, distributed_contiguous_range R, distributed_contiguous_range O>
void
inclusive_scan(ExecutionPolicy&& policy, R&& r, O&& o)
{
    inclusive_scan(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), std::forward<O>(o),
                   std::plus<stdrng::range_value_t<R>>());
}

// Distributed iterator versions

template <typename ExecutionPolicy, distributed_iterator Iter, distributed_iterator OutputIter, typename BinaryOp,
          typename T>
OutputIter
inclusive_scan(ExecutionPolicy&& policy, Iter first, Iter last, OutputIter d_first, BinaryOp binary_op, T init)
{
    auto dist = stdrng::distance(first, last);
    auto d_last = d_first;
    stdrng::advance(d_last, dist);
    inclusive_scan(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last),
                   stdrng::subrange(d_first, d_last), binary_op, init);

    return d_last;
}

template <typename ExecutionPolicy, distributed_iterator Iter, distributed_iterator OutputIter, typename BinaryOp>
OutputIter
inclusive_scan(ExecutionPolicy&& policy, Iter first, Iter last, OutputIter d_first, BinaryOp binary_op)
{
    auto dist = stdrng::distance(first, last);
    auto d_last = d_first;
    stdrng::advance(d_last, dist);
    inclusive_scan(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last),
                   stdrng::subrange(d_first, d_last), binary_op);

    return d_last;
}

template <typename ExecutionPolicy, distributed_iterator Iter, distributed_iterator OutputIter>
OutputIter
inclusive_scan(ExecutionPolicy&& policy, Iter first, Iter last, OutputIter d_first)
{
    auto dist = stdrng::distance(first, last);
    auto d_last = d_first;
    stdrng::advance(d_last, dist);
    inclusive_scan(std::forward<ExecutionPolicy>(policy), stdrng::subrange(first, last),
                   stdrng::subrange(d_first, d_last));

    return d_last;
}

// Execution policy-less versions

template <distributed_contiguous_range R, distributed_contiguous_range O>
void
inclusive_scan(R&& r, O&& o)
{
    inclusive_scan(par_unseq, std::forward<R>(r), std::forward<O>(o));
}

template <distributed_contiguous_range R, distributed_contiguous_range O, typename BinaryOp>
void
inclusive_scan(R&& r, O&& o, BinaryOp binary_op)
{
    inclusive_scan(par_unseq, std::forward<R>(r), std::forward<O>(o), binary_op);
}

template <distributed_contiguous_range R, distributed_contiguous_range O, typename BinaryOp, typename T>
void
inclusive_scan(R&& r, O&& o, BinaryOp binary_op, T init)
{
    inclusive_scan(par_unseq, std::forward<R>(r), std::forward<O>(o), binary_op, init);
}

// Distributed iterator versions

template <distributed_iterator Iter, distributed_iterator OutputIter>
OutputIter
inclusive_scan(Iter first, Iter last, OutputIter d_first)
{
    return inclusive_scan(par_unseq, first, last, d_first);
}

template <distributed_iterator Iter, distributed_iterator OutputIter, typename BinaryOp>
OutputIter
inclusive_scan(Iter first, Iter last, OutputIter d_first, BinaryOp binary_op)
{
    return inclusive_scan(par_unseq, first, last, d_first, binary_op);
}

template <distributed_iterator Iter, distributed_iterator OutputIter, typename BinaryOp, typename T>
OutputIter
inclusive_scan(Iter first, Iter last, OutputIter d_first, BinaryOp binary_op, T init)
{
    return inclusive_scan(par_unseq, first, last, d_first, binary_op, init);
}

} // namespace oneapi::dpl::experimental::dr::sp
