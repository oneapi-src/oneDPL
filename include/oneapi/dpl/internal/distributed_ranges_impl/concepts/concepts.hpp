// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges.hpp>

namespace dr {

template <typename I>
concept remote_iterator =
    std::forward_iterator<I> && requires(I &iter) { dr::ranges::rank(iter); };

template <typename R>
concept remote_range =
    rng::forward_range<R> && requires(R &r) { dr::ranges::rank(r); };

template <typename R>
concept distributed_range =
    rng::forward_range<R> && requires(R &r) { dr::ranges::segments(r); };

template <typename I>
concept remote_contiguous_iterator =
    std::random_access_iterator<I> && requires(I &iter) {
      dr::ranges::rank(iter);
      { dr::ranges::local(iter) } -> std::contiguous_iterator;
    };

template <typename I>
concept distributed_iterator = std::forward_iterator<I> && requires(I &iter) {
  dr::ranges::segments(iter);
};

template <typename R>
concept remote_contiguous_range =
    remote_range<R> && rng::random_access_range<R> && requires(R &r) {
      { dr::ranges::local(r) } -> rng::contiguous_range;
    };

template <typename R>
concept distributed_contiguous_range =
    distributed_range<R> && rng::random_access_range<R> &&
    requires(R &r) {
      { dr::ranges::segments(r) } -> rng::random_access_range;
    } &&
    remote_contiguous_range<
        rng::range_value_t<decltype(dr::ranges::segments(std::declval<R>()))>>;

template <typename Iter>
concept distributed_contiguous_iterator =
    distributed_iterator<Iter> && rng::random_access_iterator<Iter> &&
    requires(Iter &iter) {
      { dr::ranges::segments(iter) } -> rng::random_access_range;
    } &&
    remote_contiguous_range<rng::range_value_t<decltype(dr::ranges::segments(
        std::declval<Iter>()))>>;

} // namespace dr
