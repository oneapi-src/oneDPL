// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/standard_views.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/iota.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/views.hpp>

namespace experimental::shp::views {

inline constexpr auto all = rng::views::all;

inline constexpr auto counted = rng::views::counted;

inline constexpr auto drop = rng::views::drop;

inline constexpr auto iota = dr::views::iota;

inline constexpr auto take = rng::views::take;

inline constexpr auto transform = dr::views::transform;

} // namespace experimental::shp::views
