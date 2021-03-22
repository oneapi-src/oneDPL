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
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int rotate_val = 6;

    using namespace oneapi::dpl::experimental::ranges;

    auto view1 = nano::ranges::views::all(data) | views::rotate(rotate_val);
    auto view2 = views::rotate(nano::ranges::views::all(data), rotate_val);

    //check result
    ::std::rotate_copy(data, data + rotate_val, data + max_n, expected);

    EXPECT_EQ_N(view1.begin(), expected, max_n, "wrong result from rotate adapter, a pipe call");
    EXPECT_EQ_N(view2.begin(), expected, max_n, "wrong result from rotate adapter, a single CPO call");

#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
