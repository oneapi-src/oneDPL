// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/standard_views.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/iota.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/views.hpp>

<<<<<<< HEAD
namespace experimental::dr::shp::views {
=======
namespace experimental::shp::views {
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0

inline constexpr auto all = rng::views::all;

inline constexpr auto counted = rng::views::counted;

inline constexpr auto drop = rng::views::drop;

inline constexpr auto iota = experimental::dr::views::iota;

inline constexpr auto take = rng::views::take;

inline constexpr auto transform = experimental::dr::views::transform;

<<<<<<< HEAD
} // namespace experimental::dr::shp::views
=======
} // namespace experimental::shp::views
>>>>>>> cd565891f4ffdd0b4641810a38c60c683e5f1fe0
