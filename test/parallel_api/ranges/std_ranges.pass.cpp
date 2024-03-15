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

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    auto f_mutuable = [](auto&& val) -> decltype(auto) { return val *= val; };
    auto proj_mutuable = [](auto&& val) -> decltype(auto) { return val *= 2; };

    auto f = [](auto&& val) -> decltype(auto) { return val * val; };
    auto proj = [](auto&& val) -> decltype(auto) { return val * 2; };
    auto pred = [](auto&& val) -> decltype(auto) { return val == 5; };

    test_range_algo<1>{}(oneapi::dpl::ranges::for_each, std::ranges::for_each, f_mutuable, proj_mutuable);
    test_range_algo<2>{}(oneapi::dpl::ranges::transform, std::ranges::transform, f, proj);

    test_range_algo<1>{}(oneapi::dpl::ranges::find_if, std::ranges::find_if, pred, proj);
    test_range_algo<1>{}(oneapi::dpl::ranges::find_if_not, std::ranges::find_if_not, pred, proj);
    test_range_algo<1>{}(oneapi::dpl::ranges::find, std::ranges::find, 4, proj);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
