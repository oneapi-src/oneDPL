// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/span.hpp>
#include <span>

namespace oneapi::dpl::experimental::dr::shp
{

// A `remote_span` is simply a normal `std::span` that's
// been decorated with an extra `rank()` function, showing
// which rank its memory is located on.
// (Thus fulfilling the `remote_range` concept.)
/*
template <class T,
          std::size_t Extent = std::dynamic_extent>
class remote_span : public std::span<T, Extent> {
public:
  constexpr remote_span() noexcept {}

  template< class It >
  explicit(Extent != std::dynamic_extent)
  constexpr remote_span(It first, std::size_t count, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, count) {}

  template< class It, class End >
  explicit(Extent != std::dynamic_extent)
  constexpr remote_span(It first, End last, std::size_t rank)
    : rank_(rank), std::span<T, Extent>(first, last) {}

  constexpr std::size_t rank() const noexcept {
    return rank_;
  }

private:
  std::size_t rank_;
};
*/

template <typename T, typename Iter = T*>
class remote_span : public span<T, Iter>
{
  public:
    constexpr remote_span() noexcept {}

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::size_t;
    using reference = std::iter_reference_t<Iter>;

    template <rng::random_access_range R>
    requires(remote_range<R>) remote_span(R&& r) : span<T, Iter>(rng::begin(r), rng::size(r)), rank_(ranges::rank(r))
    {
    }

    template <rng::random_access_range R>
    remote_span(R&& r, std::size_t rank) : span<T, Iter>(rng::begin(r), rng::size(r)), rank_(rank)
    {
    }

    template <class It>
    constexpr remote_span(It first, std::size_t count, std::size_t rank) : span<T, Iter>(first, count), rank_(rank)
    {
    }

    template <class It, class End>
    constexpr remote_span(It first, End last, std::size_t rank) : span<T, Iter>(first, last), rank_(rank)
    {
    }

    constexpr std::size_t
    rank() const noexcept
    {
        return rank_;
    }

    remote_span
    first(std::size_t n) const
    {
        return remote_span(this->begin(), this->begin() + n, rank_);
    }

    remote_span
    last(std::size_t n) const
    {
        return remote_span(this->end() - n, this->end(), rank_);
    }

    remote_span
    subspan(std::size_t offset, std::size_t count) const
    {
        return remote_span(this->begin() + offset, this->begin() + offset + count, rank_);
    }

  private:
    std::size_t rank_;
};

template <rng::random_access_range R>
remote_span(R&&) -> remote_span<rng::range_value_t<R>, rng::iterator_t<R>>;

template <rng::random_access_range R>
remote_span(R&&, std::size_t) -> remote_span<rng::range_value_t<R>, rng::iterator_t<R>>;

} // namespace oneapi::dpl::experimental::dr::shp
