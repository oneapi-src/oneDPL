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

#include <limits>

#include "../../concepts/concepts.hpp"
#include "../../detail/std_ranges_shim.hpp"
#include "../../views/iota.hpp"
#include "for_each.hpp"

namespace oneapi::dpl::experimental::dr::sp
{

template <distributed_range R, std::integral T>
void
iota(R&& r, T value)
{
    auto iota_view = stdrng::views::iota(value, T(value + stdrng::distance(r)));

    for_each(par_unseq, views::zip(iota_view, r), [](auto&& elem) {
        auto&& [idx, v] = elem;
        v = idx;
    });
}

template <distributed_iterator Iter, std::integral T>
void
iota(Iter begin, Iter end, T value)
{
    auto r = stdrng::subrange(begin, end);
    iota(r, value);
}

} // namespace oneapi::dpl::experimental::dr::sp
