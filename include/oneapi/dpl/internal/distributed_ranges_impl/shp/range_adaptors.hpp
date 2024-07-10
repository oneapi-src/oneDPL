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
#include <oneapi/dpl/internal/distributed_ranges_impl/shp/zip_view.hpp>

namespace oneapi::dpl::experimental::dr::shp
{

template <rng::range R>
auto
enumerate(R&& r)
{
    auto i = rng::views::iota(uint32_t(0), uint32_t(rng::size(r)));
    return zip_view(i, r);
}

} // namespace oneapi::dpl::experimental::dr::shp
