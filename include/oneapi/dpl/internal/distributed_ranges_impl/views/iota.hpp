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

namespace oneapi::dpl::experimental::dr::views
{

//
// range-v3 iota uses sentinels that are not the same type as the
// iterator. A zip that uses an iota has the same issue. Make our own.
//

struct iota_fn_
{
    template <std::integral W>
    auto
    operator()(W value) const
    {
        return rng::views::iota(value, std::numeric_limits<W>::max());
    }

    template <std::integral W, std::integral Bound>
    auto
    operator()(W value, Bound bound) const
    {
        return rng::views::iota(value, W(bound));
    }
};

inline constexpr auto iota = iota_fn_{};

} // namespace oneapi::dpl::experimental::dr::views
