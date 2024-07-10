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

namespace oneapi::dpl::experimental::dr::__detail
{

inline std::size_t
round_up(std::size_t n, std::size_t multiple)
{
    if (multiple == 0)
    {
        return n;
    }

    int remainder = n % multiple;
    if (remainder == 0)
    {
        return n;
    }

    return n + multiple - remainder;
}

inline std::size_t
partition_up(std::size_t n, std::size_t multiple)
{
    if (multiple == 0)
    {
        return n;
    }

    return round_up(n, multiple) / multiple;
}

} // namespace oneapi::dpl::experimental::dr::__detail
