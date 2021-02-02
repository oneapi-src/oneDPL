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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#if _ENABLE_RANGES_TESTING
#include _PSTL_TEST_HEADER(ranges)
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n];

    auto lambda1 = [](auto i) { return i * i; };
    auto lambda2 = [](auto i, auto j) { return i + j; };

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        auto view = iota_view(0, max_n) | views::transform(lambda1);

        auto range_res = all_view<int, sycl::access::mode::write>(B);
        transform(TestUtils::default_dpcpp_policy, view, view, range_res, lambda2);
    }

    //check result
    int expected[max_n];
    ::std::transform(data, data + max_n, expected, lambda1);
    ::std::transform(expected, expected + max_n, expected, expected, lambda2);

    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from trasnform2 with sycl ranges");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
