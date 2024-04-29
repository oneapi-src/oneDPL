// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/concepts/concepts.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>

namespace oneapi::dpl::experimental::dr {

#if (defined _cpp_lib_ranges_zip)
// returns range: [(rank, element) ...]
auto ranked_view(const distributed_range auto &r) {
  auto rank = [](auto &&v) { return ranges::rank(&v); };
  return rng::views::zip(rng::views::transform(r, rank), r);
}
#endif

} // namespace oneapi::dpl::experimental::dr
