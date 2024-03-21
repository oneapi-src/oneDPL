// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/standard_views.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/zip_view.hpp>

namespace experimental::dr::shp {

template <rng::range R> auto enumerate(R &&r) {
  auto i = rng::views::iota(uint32_t(0), uint32_t(rng::size(r)));
  return experimental::dr::shp::zip_view(i, r);
}

} // namespace experimental::dr::shp
