// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <sycl/sycl.hpp>

#include <dr/detail/sycl_utils.hpp>
#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/detail.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/util.hpp>
#include <dr/shp/zip_view.hpp>

namespace dr::shp {

template <typename ExecutionPolicy, dr::distributed_range R, typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {
  static_assert( // currently only one policy supported
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  std::vector<sycl::event> events;

  for (auto &&segment : dr::ranges::segments(r)) {
    auto &&q = __detail::queue(dr::ranges::rank(segment));

    assert(rng::distance(segment) > 0);

    auto local_segment = __detail::local(segment);

    auto first = rng::begin(local_segment);

    auto event = dr::__detail::parallel_for(
        q, sycl::range<>(rng::distance(local_segment)),
        [=](auto idx) { fn(*(first + idx)); });
    events.emplace_back(event);
  }
  __detail::wait(events);
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter, typename Fn>
void for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn) {
  for_each(std::forward<ExecutionPolicy>(policy), rng::subrange(begin, end),
           std::forward<Fn>(fn));
}

template <dr::distributed_range R, typename Fn> void for_each(R &&r, Fn &&fn) {
  for_each(dr::shp::par_unseq, std::forward<R>(r), std::forward<Fn>(fn));
}

template <dr::distributed_iterator Iter, typename Fn>
void for_each(Iter begin, Iter end, Fn &&fn) {
  for_each(dr::shp::par_unseq, begin, end, std::forward<Fn>(fn));
}

} // namespace dr::shp
