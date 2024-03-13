// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>

#include <dr/detail/ranges_shim.hpp>

namespace dr::shp {

template <typename T, rng::random_access_iterator Iter = T *>
class span : public rng::view_interface<dr::shp::span<T, Iter>> {
public:
  static_assert(std::is_same_v<std::iter_value_t<Iter>, T>);

  using value_type = std::iter_value_t<Iter>;
  using size_type = std::size_t;
  using difference_type = std::iter_difference_t<Iter>;
  using reference = std::iter_reference_t<Iter>;
  using iterator = Iter;

  template <rng::random_access_range R>
  span(R &&r) : begin_(rng::begin(r)), end_(rng::end(r)) {}
  span(Iter first, Iter last) : begin_(first), end_(last) {}
  span(Iter first, std::size_t count) : begin_(first), end_(first + count) {}

  span() = default;
  span(const span &) noexcept = default;
  span &operator=(const span &) noexcept = default;

  std::size_t size() const noexcept { return std::size_t(end() - begin()); }

  bool empty() const noexcept { return size() == 0; }

  Iter begin() const noexcept { return begin_; }

  Iter end() const noexcept { return end_; }

  reference operator[](size_type index) const { return *(begin() + index); }

  span first(size_type n) const { return span(begin(), begin() + n); }

  span last(size_type n) const { return span(end() - n, end()); }

  span subspan(std::size_t offset, std::size_t count) const {
    return span(begin() + offset, begin() + offset + count);
  }

private:
  Iter begin_, end_;
};

template <rng::random_access_range R>
span(R &&) -> span<rng::range_value_t<R>, rng::iterator_t<R>>;

template <rng::random_access_iterator Iter>
span(Iter first, std::size_t count) -> span<std::iter_value_t<Iter>, Iter>;

} // namespace dr::shp
