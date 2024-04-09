// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/detail/onedpl_direct_iterator.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/algorithms/execution_policy.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/init.hpp>
#include <sycl/sycl.hpp>

namespace {

// Precondition: rng::distance(first, last) >= 2
// Postcondition: return future to [first, last) reduced with fn
template <typename T, typename ExecutionPolicy,
          std::bidirectional_iterator Iter, typename Fn>
auto reduce_no_init_async(ExecutionPolicy &&policy, Iter first, Iter last,
                          Fn &&fn) {
  Iter new_last = last;
  --new_last;

  std::iter_value_t<Iter> init = *new_last;

  oneapi::dpl::experimental::dr::__detail::direct_iterator d_first(first);
  oneapi::dpl::experimental::dr::__detail::direct_iterator d_last(new_last);

  return oneapi::dpl::experimental::reduce_async(
      std::forward<ExecutionPolicy>(policy), d_first, d_last,
      static_cast<T>(init), std::forward<Fn>(fn));
}

template <typename T, typename ExecutionPolicy,
          std::bidirectional_iterator Iter, typename Fn>
  requires(sycl::has_known_identity_v<Fn, T>)
auto reduce_no_init_async(ExecutionPolicy &&policy, Iter first, Iter last,
                          Fn &&fn) {
  oneapi::dpl::experimental::dr::__detail::direct_iterator d_first(first);
  oneapi::dpl::experimental::dr::__detail::direct_iterator d_last(last);

  return oneapi::dpl::experimental::reduce_async(
      std::forward<ExecutionPolicy>(policy), d_first, d_last,
      sycl::known_identity_v<Fn, T>, std::forward<Fn>(fn));
}

} // namespace

namespace oneapi::dpl::experimental::dr::shp {

template <typename ExecutionPolicy, distributed_range R, typename T,
          typename BinaryOp>
T reduce(ExecutionPolicy &&policy, R &&r, T init, BinaryOp &&binary_op) {

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    using future_t = decltype(reduce_async(
        __detail::dpl_policy(0), ranges::segments(r)[0].begin(),
        ranges::segments(r)[0].end(), init, binary_op));

    std::vector<future_t> futures;

    for (auto &&segment : ranges::segments(r)) {
      auto &&local_policy = __detail::dpl_policy(ranges::rank(segment));

      auto dist = rng::distance(segment);
      if (dist <= 0) {
        continue;
      } else if (dist == 1) {
        init = binary_op(init, *rng::begin(segment));
        continue;
      }

      auto future = reduce_no_init_async<T>(local_policy, rng::begin(segment),
                                            rng::end(segment), binary_op);

      futures.push_back(std::move(future));
    }

    for (auto &&f : futures) {
      init = binary_op(init, f.get());
    }

    return init;
  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, distributed_range R, typename T>
T reduce(ExecutionPolicy &&policy, R &&r, T init) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), init,
                std::plus<>());
}

template <typename ExecutionPolicy, distributed_range R>
rng::range_value_t<R> reduce(ExecutionPolicy &&policy, R &&r) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r),
                rng::range_value_t<R>{}, std::plus<>());
}

// Iterator versions

template <typename ExecutionPolicy, distributed_iterator Iter>
std::iter_value_t<Iter> reduce(ExecutionPolicy &&policy, Iter first,
                               Iter last) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(first, last), std::iter_value_t<Iter>{},
                std::plus<>());
}

template <typename ExecutionPolicy, distributed_iterator Iter, typename T>
T reduce(ExecutionPolicy &&policy, Iter first, Iter last, T init) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(first, last), init, std::plus<>());
}

template <typename ExecutionPolicy, distributed_iterator Iter, typename T,
          typename BinaryOp>
T reduce(ExecutionPolicy &&policy, Iter first, Iter last, T init,
         BinaryOp &&binary_op) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(first, last), init,
                std::forward<BinaryOp>(binary_op));
}

// Execution policy-less algorithms

template <distributed_range R> rng::range_value_t<R> reduce(R &&r) {
  return reduce(par_unseq, std::forward<R>(r));
}

template <distributed_range R, typename T> T reduce(R &&r, T init) {
  return reduce(par_unseq, std::forward<R>(r), init);
}

template <distributed_range R, typename T, typename BinaryOp>
T reduce(R &&r, T init, BinaryOp &&binary_op) {
  return reduce(par_unseq, std::forward<R>(r), init,
                std::forward<BinaryOp>(binary_op));
}

template <distributed_iterator Iter>
std::iter_value_t<Iter> reduce(Iter first, Iter last) {
  return reduce(par_unseq, first, last);
}

template <distributed_iterator Iter, typename T>
T reduce(Iter first, Iter last, T init) {
  return reduce(par_unseq, first, last, init);
}

template <distributed_iterator Iter, typename T, typename BinaryOp>
T reduce(Iter first, Iter last, T init, BinaryOp &&binary_op) {
  return reduce(par_unseq, first, last, init,
                std::forward<BinaryOp>(binary_op));
}

} // namespace oneapi::dpl::experimental::dr::shp
