// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/detail/enumerate.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/remote_subrange.hpp>
#include <dr/detail/view_detectors.hpp>

namespace dr {

namespace __detail {

// Take all elements up to and including segment `segment_id` at index
// `local_id`
template <typename R>
auto take_segments(R &&segments, std::size_t last_seg, std::size_t local_id) {
  auto remainder = local_id;

  auto take_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    if (i == last_seg) {
      auto first = rng::begin(segment);
      auto last = rng::begin(segment);
      rng::advance(last, remainder);
      return dr::remote_subrange(first, last, dr::ranges::rank(segment));
    } else {
      return dr::remote_subrange(segment);
    }
  };

  return enumerate(segments) | rng::views::take(last_seg + 1) |
         rng::views::transform(std::move(take_partial));
}

// Take the first n elements
template <typename R> auto take_segments(R &&segments, std::size_t n) {
  std::size_t last_seg = 0;
  std::size_t remainder = n;

  for (auto &&seg : segments) {
    if (seg.size() >= remainder) {
      break;
    }
    remainder -= seg.size();
    last_seg++;
  }

  return take_segments(std::forward<R>(segments), last_seg, remainder);
}

// Drop all elements up to segment `segment_id` and index `local_id`
template <typename R>
auto drop_segments(R &&segments, std::size_t first_seg, std::size_t local_id) {
  auto remainder = local_id;

  auto drop_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    if (i == first_seg) {
      auto first = rng::begin(segment);
      rng::advance(first, remainder);
      auto last = rng::end(segment);
      return dr::remote_subrange(first, last, dr::ranges::rank(segment));
    } else {
      return dr::remote_subrange(segment);
    }
  };

  return enumerate(segments) | rng::views::drop(first_seg) |
         rng::views::transform(std::move(drop_partial));
}

// Drop the first n elements
template <typename R> auto drop_segments(R &&segments, std::size_t n) {
  std::size_t first_seg = 0;
  std::size_t remainder = n;

  for (auto &&seg : segments) {
    if (seg.size() > remainder) {
      break;
    }
    remainder -= seg.size();
    first_seg++;
  }

  return drop_segments(std::forward<R>(segments), first_seg, remainder);
}

} // namespace __detail

} // namespace dr

namespace DR_RANGES_NAMESPACE {

// A standard library range adaptor does not change the rank of a
// remote range, so we can simply return the rank of the base view.
template <rng::range V>
  requires(dr::remote_range<decltype(std::declval<V>().base())>)
auto rank_(V &&v) {
  return dr::ranges::rank(std::forward<V>(v).base());
}

template <rng::range V>
  requires(dr::is_ref_view_v<std::remove_cvref_t<V>> &&
           dr::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return dr::ranges::segments(v.base());
}

template <rng::range V>
  requires(dr::is_take_view_v<std::remove_cvref_t<V>> &&
           dr::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return dr::__detail::take_segments(dr::ranges::segments(v.base()), v.size());
}

template <rng::range V>
  requires(dr::is_drop_view_v<std::remove_cvref_t<V>> &&
           dr::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return dr::__detail::drop_segments(dr::ranges::segments(v.base()),
                                     v.base().size() - v.size());
}

template <rng::range V>
  requires(dr::is_subrange_view_v<std::remove_cvref_t<V>> &&
           dr::distributed_iterator<rng::iterator_t<V>>)
auto segments_(V &&v) {
  auto first = rng::begin(v);
  auto last = rng::end(v);
  auto size = rng::distance(first, last);

  return dr::__detail::take_segments(dr::ranges::segments(first), size);
}

} // namespace DR_RANGES_NAMESPACE
