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

#include <cmath>

namespace oneapi::dpl::experimental::dr::sp
{

namespace detail
{

// Factor n into 2 roughly equal factors
// n = pq, p >= q
inline std::tuple<std::size_t, std::size_t>
factor(std::size_t n)
{
    std::size_t q = std::sqrt(n);

    while (q > 1 && n / q != static_cast<double>(n) / q)
    {
        q--;
    }
    std::size_t p = n / q;

    return {p, q};
}

} // namespace detail

} // namespace oneapi::dpl::experimental::dr::sp
