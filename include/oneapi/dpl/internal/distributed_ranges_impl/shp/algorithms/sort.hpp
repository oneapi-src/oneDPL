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

#include <oneapi/dpl/execution>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/onedpl_direct_iterator.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/init.hpp>

#include <omp.h>
#include <sycl/sycl.hpp>

namespace oneapi::dpl::experimental::dr::shp
{

namespace __detail
{

template <typename LocalPolicy, typename InputIt, typename Compare>
sycl::event
sort_async(LocalPolicy&& policy, InputIt first, InputIt last, Compare comp)
{
    if (rng::distance(first, last) >= 2)
    {
        dr::__detail::direct_iterator d_first(first);
        dr::__detail::direct_iterator d_last(last);
        return oneapi::dpl::experimental::sort_async(std::forward<LocalPolicy>(policy), d_first, d_last, comp);
    }
    else
    {
        return sycl::event{};
    }
}

template <typename LocalPolicy, typename InputIt1, typename InputIt2, typename OutputIt,
          typename Comparator = std::less<>>
OutputIt
lower_bound(LocalPolicy&& policy, InputIt1 start, InputIt1 end, InputIt2 value_first, InputIt2 value_last,
            OutputIt result, Comparator comp = Comparator())
{
    dr::__detail::direct_iterator d_start(start);
    dr::__detail::direct_iterator d_end(end);

    dr::__detail::direct_iterator d_value_first(value_first);
    dr::__detail::direct_iterator d_value_last(value_last);

    dr::__detail::direct_iterator d_result(result);

    return oneapi::dpl::lower_bound(std::forward<LocalPolicy>(policy), d_start, d_end, d_value_first, d_value_last,
                                    d_result, comp)
        .base();
}

} // namespace __detail

template <distributed_range R, typename Compare = std::less<>>
void
sort(R&& r, Compare comp = Compare())
{
    auto&& segments = ranges::segments(r);

    if (rng::size(segments) == 0)
    {
        return;
    }
    else if (rng::size(segments) == 1)
    {
        auto&& segment = *rng::begin(segments);
        auto&& local_policy = __detail::dpl_policy(ranges::rank(segment));
        auto&& local_segment = __detail::local(segment);

        __detail::sort_async(local_policy, rng::begin(local_segment), rng::end(local_segment), comp).wait();
        return;
    }

    using T = rng::range_value_t<R>;
    std::vector<sycl::event> events;

    const std::size_t n_segments = std::size_t(rng::size(segments));
    const std::size_t n_splitters = n_segments - 1;

    // Sort each local segment, then compute medians.
    // Each segment has `n_splitters` medians,
    // so `n_segments * n_splitters` medians total.

    T* medians = sycl::malloc_device<T>(n_segments * n_splitters, shp::devices()[0], shp::context());

    for (auto&& [segment_id_, segment] : rng::views::enumerate(segments))
    {
        auto const segment_id = static_cast<std::size_t>(segment_id_);
        auto&& q = __detail::queue(ranges::rank(segment));
        auto&& local_policy = __detail::dpl_policy(ranges::rank(segment));

        auto&& local_segment = __detail::local(segment);

        auto s = __detail::sort_async(local_policy, rng::begin(local_segment), rng::end(local_segment), comp);

        double step_size = static_cast<double>(rng::size(segment)) / n_segments;

        auto local_begin = rng::begin(local_segment);

        auto e = q.submit([&](auto&& h) {
            h.depends_on(s);

            h.parallel_for(n_splitters, [=](auto i) {
                medians[n_splitters * segment_id + i] = local_begin[std::size_t(step_size * (i + 1) + 0.5)];
            });
        });

        events.push_back(e);
    }

    __detail::wait(events);
    events.clear();

    // Compute global medians by sorting medians and
    // computing `n_splitters` medians from the medians.
    auto&& local_policy = __detail::dpl_policy(0);
    __detail::sort_async(local_policy, medians, medians + n_segments * n_splitters, comp).wait();

    double step_size = static_cast<double>(n_segments * n_splitters) / n_segments;

    // - Collect median of medians to get final splitters.
    // - Write splitters to [0, n_splitters) in `medians`

    auto&& q = __detail::queue(0);
    q.single_task([=] {
         for (std::size_t i = 0; i < n_splitters; i++)
         {
             medians[i] = medians[std::size_t(step_size * (i + 1) + 0.5)];
         }
     }).wait();

    std::vector<std::size_t*> splitter_indices;
    // sorted_seg_sizes[i]: how many elements exists in all segments between
    // medians[i-1] and medians[i]
    std::vector<std::size_t> sorted_seg_sizes(n_segments, 0);
    // push_positions[snd_idx][rcv_idx]: shift inside final segment of rcv_idx for
    // data being sent from initial snd_idx segment
    std::vector<std::vector<std::size_t>> push_positions(n_segments);

    // Compute how many elements will be sent to each of the new "sorted
    // segments". Simultaneously compute the offsets `push_positions` where each
    // segments' corresponding elements will be pushed.

    for (auto&& [segment_id, segment] : rng::views::enumerate(segments))
    {
        auto&& q = __detail::queue(ranges::rank(segment));
        auto&& local_policy = __detail::dpl_policy(ranges::rank(segment));

        auto&& local_segment = __detail::local(segment);

        // splitter_i = [ index in local_segment of first element greater or equal
        // 1st global median, index ... 2nd global median, ..., size of
        // local_segment]
        std::size_t* splitter_i = sycl::malloc_shared<std::size_t>(n_segments, q.get_device(), shp::context());
        splitter_indices.push_back(splitter_i);

        // Local copy `medians_l` necessary due to https://github.com/oneapi-src/distributed-ranges/issues/777
        T* medians_l = sycl::malloc_device<T>(n_splitters, q.get_device(), shp::context());

        q.memcpy(medians_l, medians, sizeof(T) * n_splitters).wait();

        __detail::lower_bound(local_policy, rng::begin(local_segment), rng::end(local_segment), medians_l,
                              medians_l + n_splitters, splitter_i, comp);

        sycl::free(medians_l, shp::context());

        splitter_i[n_splitters] = rng::size(local_segment);

        for (std::size_t i = 0; i < n_segments; i++)
        {
            const std::size_t n_elements = splitter_i[i] - (i == 0 ? 0 : splitter_i[i - 1]);
            const std::size_t pos = std::atomic_ref(sorted_seg_sizes[i]).fetch_add(n_elements);
            push_positions[static_cast<std::size_t>(segment_id)].push_back(pos);
        }
    }

    // Allocate new "sorted segments"
    std::vector<T*> sorted_segments;

    for (auto&& [segment_id, segment] : rng::views::enumerate(segments))
    {
        auto&& q = __detail::queue(ranges::rank(segment));

        T* buffer = sycl::malloc_device<T>(sorted_seg_sizes[static_cast<std::size_t>(segment_id)], q);
        sorted_segments.push_back(buffer);
    }

    // Copy corresponding elements to each "sorted segment"
    for (auto&& [segment_id_, segment] : rng::views::enumerate(segments))
    {
        auto&& local_segment = __detail::local(segment);
        const auto segment_id = static_cast<std::size_t>(segment_id_);

        std::size_t* splitter_i = splitter_indices[segment_id];

        auto p_first = rng::begin(local_segment);
        auto p_last = p_first;
        for (std::size_t i = 0; i < n_segments; i++)
        {
            p_last = rng::begin(local_segment) + splitter_i[i];

            const std::size_t pos = push_positions[segment_id][i];

            auto e = shp::copy_async(p_first, p_last, sorted_segments[i] + pos);
            events.push_back(e);

            p_first = p_last;
        }
    }

    __detail::wait(events);
    events.clear();

    // merge sorted chunks within each of these new segments

#pragma omp parallel num_threads(n_segments)
    {
        int t = omp_get_thread_num();

        std::vector<std::size_t> chunks_ind;
        for (std::size_t i = 0; i < n_segments; i++)
        {
            chunks_ind.push_back(push_positions[i][t]);
        }

        auto _segments = n_segments;
        while (_segments > 1)
        {
            std::vector<std::size_t> new_chunks;
            new_chunks.push_back(0);

            for (int s = 0; s < _segments / 2; s++)
            {

                const std::size_t l = (2 * s + 2 < _segments) ? chunks_ind[2 * s + 2] : sorted_seg_sizes[t];

                auto first = dr::__detail::direct_iterator(sorted_segments[t] + chunks_ind[2 * s]);
                auto middle = dr::__detail::direct_iterator(sorted_segments[t] + chunks_ind[2 * s + 1]);
                auto last = dr::__detail::direct_iterator(sorted_segments[t] + l);

                new_chunks.push_back(l);

                oneapi::dpl::inplace_merge(__detail::dpl_policy(ranges::rank(segments[t])), first, middle, last, comp);
            }

            _segments = (_segments + 1) / 2;

            std::swap(chunks_ind, new_chunks);
        }
    } // End of omp parallel region

    // Copy the results into the output.

    auto d_first = rng::begin(r);

    for (std::size_t i = 0; i < sorted_segments.size(); i++)
    {
        T* seg = sorted_segments[i];
        std::size_t n_elements = sorted_seg_sizes[i];

        auto e = shp::copy_async(seg, seg + n_elements, d_first);

        events.push_back(e);

        rng::advance(d_first, n_elements);
    }

    __detail::wait(events);

    // Free temporary memory.

    for (auto&& sorted_seg : sorted_segments)
    {
        sycl::free(sorted_seg, shp::context());
    }

    for (auto&& splitter_i : splitter_indices)
    {
        sycl::free(splitter_i, shp::context());
    }

    sycl::free(medians, shp::context());
}

template <distributed_iterator RandomIt, typename Compare = std::less<>>
void
sort(RandomIt first, RandomIt last, Compare comp = Compare())
{
    sort(rng::subrange(first, last), comp);
}

} // namespace oneapi::dpl::experimental::dr::shp
