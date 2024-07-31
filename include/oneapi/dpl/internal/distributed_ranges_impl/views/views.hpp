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

#ifndef _ONEDPL_DR_VIEWS_VIEWS_HPP
#define _ONEDPL_DR_VIEWS_VIEWS_HPP

#include "../concepts/concepts.hpp"
#include "transform.hpp"

namespace oneapi::dpl::experimental::dr
{

#if (defined _cpp_lib_ranges_zip)
// returns range: [(rank, element) ...]
auto
ranked_view(const distributed_range auto& r)
{
    auto rank = [](auto&& v) { return ranges::rank(&v); };
    return stdrng::views::zip(stdrng::views::transform(r, rank), r);
}
#endif

} // namespace oneapi::dpl::experimental::dr

#endif /* _ONEDPL_DR_VIEWS_VIEWS_HPP */
