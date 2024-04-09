// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>

namespace oneapi::dpl::experimental::dr {

// returns range: [(rank, element) ...]
auto ranked_view(const oneapi::dpl::experimental::dr::distributed_range auto &r) {
  auto rank = [](auto &&v) { return oneapi::dpl::experimental::dr::ranges::rank(&v); };
  return rng::views::zip(rng::views::transform(r, rank), r);
}

} // namespace oneapi::dpl::experimental::dr
