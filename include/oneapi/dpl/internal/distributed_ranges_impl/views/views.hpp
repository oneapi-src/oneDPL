// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/views/transform.hpp>

namespace dr {

// returns range: [(rank, element) ...]
auto ranked_view(const dr::distributed_range auto &r) {
  auto rank = [](auto &&v) { return dr::ranges::rank(&v); };
  return rng::views::zip(rng::views::transform(r, rank), r);
}

} // namespace dr
