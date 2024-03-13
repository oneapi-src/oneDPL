// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/views/standard_views.hpp>
#include <dr/views/iota.hpp>
#include <dr/views/transform.hpp>
#include <dr/views/views.hpp>

namespace dr::shp::views {

inline constexpr auto all = rng::views::all;

inline constexpr auto counted = rng::views::counted;

inline constexpr auto drop = rng::views::drop;

inline constexpr auto iota = dr::views::iota;

inline constexpr auto take = rng::views::take;

inline constexpr auto transform = dr::views::transform;

} // namespace dr::shp::views
