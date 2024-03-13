// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <optional>

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/allocators.hpp>
#include <dr/shp/detail.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/vector.hpp>
#include <dr/shp/views/views.hpp>

namespace dr::shp {

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp,
          typename U = rng::range_value_t<R>>
void inclusive_scan_impl_(ExecutionPolicy &&policy, R &&r, O &&o,
                          BinaryOp &&binary_op, std::optional<U> init = {}) {
  using T = rng::range_value_t<O>;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  auto zipped_view = dr::shp::views::zip(r, o);
  auto zipped_segments = zipped_view.zipped_segments();

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {

    std::vector<sycl::event> events;

    auto root = dr::shp::devices()[0];
    dr::shp::device_allocator<T> allocator(dr::shp::context(), root);
    dr::shp::vector<T, dr::shp::device_allocator<T>> partial_sums(
        std::size_t(zipped_segments.size()), allocator);

    std::size_t segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto &&q = __detail::queue(dr::ranges::rank(in_segment));
      auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(in_segment));

      auto dist = rng::distance(in_segment);
      assert(dist > 0);

      auto first = rng::begin(in_segment);
      auto last = rng::end(in_segment);
      auto d_first = rng::begin(out_segment);

      sycl::event event;

      if (segment_id == 0 && init.has_value()) {
        event = oneapi::dpl::experimental::inclusive_scan_async(
            local_policy, dr::__detail::direct_iterator(first),
            dr::__detail::direct_iterator(last),
            dr::__detail::direct_iterator(d_first), binary_op, init.value());
      } else {
        event = oneapi::dpl::experimental::inclusive_scan_async(
            local_policy, dr::__detail::direct_iterator(first),
            dr::__detail::direct_iterator(last),
            dr::__detail::direct_iterator(d_first), binary_op);
      }

      auto dst_iter = dr::ranges::local(partial_sums).data() + segment_id;

      auto src_iter = dr::ranges::local(out_segment).data();
      rng::advance(src_iter, dist - 1);

      auto e = q.submit([&](auto &&h) {
        h.depends_on(event);
        h.single_task([=]() {
          rng::range_value_t<O> value = *src_iter;
          *dst_iter = value;
        });
      });

      events.push_back(e);

      segment_id++;
    }

    __detail::wait(events);
    events.clear();

    auto &&local_policy = __detail::dpl_policy(0);

    auto first = dr::ranges::local(partial_sums).data();
    auto last = first + partial_sums.size();

    oneapi::dpl::experimental::inclusive_scan_async(local_policy, first, last,
                                                    first, binary_op)
        .wait();

    std::size_t idx = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      if (idx > 0) {
        auto &&q = __detail::queue(dr::ranges::rank(out_segment));

        auto first = rng::begin(out_segment);
        dr::__detail::direct_iterator d_first(first);

        auto d_sum =
            dr::ranges::__detail::local(partial_sums).begin() + idx - 1;

        sycl::event e = dr::__detail::parallel_for(
            q, sycl::range<>(rng::distance(out_segment)),
            [=](auto idx) { d_first[idx] = binary_op(d_first[idx], *d_sum); });

        events.push_back(e);
      }
      idx++;
    }

    __detail::wait(events);

  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp, typename T>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o,
                    BinaryOp &&binary_op, T init) {
  inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o),
                       std::forward<BinaryOp>(binary_op), std::optional(init));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o,
                    BinaryOp &&binary_op) {
  inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o),
                       std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o) {
  inclusive_scan(std::forward<ExecutionPolicy>(policy), std::forward<R>(r),
                 std::forward<O>(o), std::plus<rng::range_value_t<R>>());
}

// Distributed iterator versions

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename BinaryOp, typename T>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first, BinaryOp &&binary_op, T init) {

  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last),
                 std::forward<BinaryOp>(binary_op), init);

  return d_last;
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename BinaryOp>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first, BinaryOp &&binary_op) {

  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last),
                 std::forward<BinaryOp>(binary_op));

  return d_last;
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last));

  return d_last;
}

// Execution policy-less versions

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O>
void inclusive_scan(R &&r, O &&o) {
  inclusive_scan(dr::shp::par_unseq, std::forward<R>(r), std::forward<O>(o));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp>
void inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op) {
  inclusive_scan(dr::shp::par_unseq, std::forward<R>(r), std::forward<O>(o),
                 std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp, typename T>
void inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op, T init) {
  inclusive_scan(dr::shp::par_unseq, std::forward<R>(r), std::forward<O>(o),
                 std::forward<BinaryOp>(binary_op), init);
}

// Distributed iterator versions

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first) {
  return inclusive_scan(dr::shp::par_unseq, first, last, d_first);
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op) {
  return inclusive_scan(dr::shp::par_unseq, first, last, d_first,
                        std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp, typename T>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op, T init) {
  return inclusive_scan(dr::shp::par_unseq, first, last, d_first,
                        std::forward<BinaryOp>(binary_op), init);
}

} // namespace dr::shp
