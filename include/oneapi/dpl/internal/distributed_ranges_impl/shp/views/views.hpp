// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <oneapi/dpl/internal/distributed_ranges_impl/shp/views/standard_views.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/iota.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/transform.hpp>
#include <oneapi/dpl/internal/distributed_ranges_impl/views/views.hpp>

namespace oneapi::dpl::experimental::dr::shp::views
{

inline constexpr auto all = rng::views::all;

inline constexpr auto counted = rng::views::counted;

inline constexpr auto drop = rng::views::drop;

inline constexpr auto iota = dr::views::iota;

inline constexpr auto take = rng::views::take;

inline constexpr auto transform = dr::views::transform;

} // namespace oneapi::dpl::experimental::dr::shp::views
