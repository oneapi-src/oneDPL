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

#include "standard_views.hpp"
#include "../algorithms/iota.hpp"
#include "../../views/transform.hpp"
#include "../../views/views.hpp"

namespace oneapi::dpl::experimental::dr::sp::views
{

inline constexpr auto all = stdrng::views::all;

inline constexpr auto counted = stdrng::views::counted;

inline constexpr auto drop = stdrng::views::drop;

inline constexpr auto iota = dr::views::iota;

inline constexpr auto take = stdrng::views::take;

inline constexpr auto transform = dr::views::transform;

} // namespace oneapi::dpl::experimental::dr::sp::views
