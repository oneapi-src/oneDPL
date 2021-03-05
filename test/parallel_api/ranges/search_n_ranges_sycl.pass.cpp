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
    const int count = 10;
    int data[count] = {0, 1, 2, 3, 4, 4, 4, 7, 8, 9};

    const int n_val = 3;
    const int idx = 4;
    const int val = data[idx];
    int res = -1;

    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(count));

        auto view_a = all_view(A);
        res = search_n(TestUtils::default_dpcpp_policy, view_a, n_val, val, [](auto a, auto b) { return a == b; });
    }

    //check result
    EXPECT_TRUE(res == idx, "wrong effect from 'search_n' with sycl ranges");

#else
    TestUtils::skip();

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
