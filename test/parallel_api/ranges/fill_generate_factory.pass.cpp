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

#include <oneapi/dpl/execution>

#include "support/pstl_test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    auto lambda = [](auto i) { return i * i; };

    using namespace oneapi::dpl::experimental::ranges;

    auto view1 = views::fill(-1, max_n) | views::transform(lambda);
    auto res1 = std::all_of(view1.begin(), view1.end(), [](auto i) { return i == 1;});

    auto view2 = views::generate([]() { return -1;}, max_n) | views::transform(lambda);
    auto res2 = std::all_of(view2.begin(), view2.end(), [](auto i) { return i == 1;});

    //check result
    EXPECT_TRUE(res1, "wrong result from fill factory");
    EXPECT_TRUE(res2, "wrong result from generate factory");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
