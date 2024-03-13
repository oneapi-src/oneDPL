// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/shp/span.hpp>
#include <span>

namespace dr::shp {

// A `device_span` is simply a normal `std::span` that's
// been decorated with an extra `rank()` function, showing
// which rank its memory is located on.
// (Thus fulfilling the `remote_range` concept.)
/*
template <class T,
          std::size_t Extent = std::dynamic_extent>
class device_span : public std::span<T, Extent> {
public:
  constexpr device_span() noexcept {}

  template< class It >
  explicit(Extent != std::dynamic_extent)
  constexpr device_span(It first, std::size_t count, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, count) {}

  template< class It, class End >
  explicit(Extent != std::dynamic_extent)
  constexpr device_span(It first, End last, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, last) {}

  constexpr std::size_t rank() const noexcept {
    return rank_;
  }

private:
  std::size_t rank_;
};
*/

template <typename T, typename Iter = T *>
class device_span : public dr::shp::span<T, Iter> {
public:
  constexpr device_span() noexcept {}

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::size_t;
  using reference = std::iter_reference_t<Iter>;

  template <rng::random_access_range R>
    requires(dr::remote_range<R>)
  device_span(R &&r)
      : dr::shp::span<T, Iter>(rng::begin(r), rng::size(r)),
        rank_(dr::ranges::rank(r)) {}

  template <rng::random_access_range R>
  device_span(R &&r, std::size_t rank)
      : dr::shp::span<T, Iter>(rng::begin(r), rng::size(r)), rank_(rank) {}

  template <class It>
  constexpr device_span(It first, std::size_t count, std::size_t rank)
      : dr::shp::span<T, Iter>(first, count), rank_(rank) {}

  template <class It, class End>
  constexpr device_span(It first, End last, std::size_t rank)
      : dr::shp::span<T, Iter>(first, last), rank_(rank) {}

  constexpr std::size_t rank() const noexcept { return rank_; }

  device_span first(std::size_t n) const {
    return device_span(this->begin(), this->begin() + n, rank_);
  }

  device_span last(std::size_t n) const {
    return device_span(this->end() - n, this->end(), rank_);
  }

  device_span subspan(std::size_t offset, std::size_t count) const {
    return device_span(this->begin() + offset, this->begin() + offset + count,
                       rank_);
  }

private:
  std::size_t rank_;
};

template <rng::random_access_range R>
device_span(R &&) -> device_span<rng::range_value_t<R>, rng::iterator_t<R>>;

template <rng::random_access_range R>
device_span(R &&, std::size_t)
    -> device_span<rng::range_value_t<R>, rng::iterator_t<R>>;

} // namespace dr::shp
