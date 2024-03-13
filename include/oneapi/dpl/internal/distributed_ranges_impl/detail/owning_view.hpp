// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>

namespace dr {

namespace __detail {

// TODO: this `owning_view` is needed because range-v3 does not have an
//       `owning_view`.  Ideally we would submit a PR to range-v3 /or
//       switch to solely using libstdc++13.

template <rng::range R>
class owning_view : public rng::view_interface<owning_view<R>> {
public:
  owning_view(R &&range) : range_(std::move(range)) {}

  owning_view()
    requires std::default_initializable<R>
  = default;
  owning_view(owning_view &&other) = default;
  owning_view(const owning_view &other) = default;

  owning_view &operator=(owning_view &&other) = default;
  owning_view &operator=(const owning_view &other) = default;

  auto size() const
    requires(rng::sized_range<R>)
  {
    return rng::size(range_);
  }

  auto empty() const
    requires(rng::sized_range<R>)
  {
    return size() == 0;
  }

  auto begin() { return rng::begin(range_); }

  auto begin() const
    requires(rng::range<const R>)
  {
    return rng::begin(range_);
  }

  auto end() { return rng::end(range_); }

  auto end() const
    requires(rng::range<const R>)
  {
    return rng::end(range_);
  }

  decltype(auto) base() { return range_; }

  decltype(auto) base() const { return range_; }

private:
  R range_;
};

} // namespace __detail

} // namespace dr
