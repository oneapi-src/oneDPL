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

#include "views/standard_views.hpp"
#include "zip_view.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

template <stdrng::range R>
auto
enumerate(R&& r)
{
    auto i = stdrng::views::iota(uint32_t(0), uint32_t(stdrng::size(r)));
    return zip_view(i, r);
}

} // namespace oneapi::dpl::experimental::dr::sp
