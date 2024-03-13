// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/views/standard_views.hpp>
#include <dr/shp/zip_view.hpp>

namespace dr::shp {

template <rng::range R> auto enumerate(R &&r) {
  auto i = rng::views::iota(uint32_t(0), uint32_t(rng::size(r)));
  return dr::shp::zip_view(i, r);
}

} // namespace dr::shp
