// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::__detail {

//
// std::ranges::enumerate handles unbounded ranges and returns a range
// where end() is a different type than begin(). Most of our code
// assumes std::ranges::common_range. bounded_enumerate requires a
// bounded range and returns a common_range.
//
template <typename R> auto bounded_enumerate(R &&r) {
  auto size = rng::distance(r);
  using W = std::uint32_t;
  return rng::views::zip(rng::views::iota(W(0), W(size)), r);
}

} // namespace dr::__detail
