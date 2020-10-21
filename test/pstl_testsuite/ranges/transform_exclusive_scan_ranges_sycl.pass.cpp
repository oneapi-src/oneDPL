// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include <iostream>

#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)
#if _PSTL_USE_RANGES
#include _PSTL_TEST_HEADER(ranges)
#endif

int32_t
main()
{
#if _PSTL_USE_RANGES
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data1[max_n];

    auto lambda = [](auto i) { return i * i; };
    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B(data1, cl::sycl::range<1>(max_n));

        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, cl::sycl::access::mode::read>(A);
        auto view_res = ranges::all_view<int, cl::sycl::access::mode::write>(B);

        ranges::transform_exclusive_scan(TestUtils::default_dpcpp_policy, view, view_res, 100, ::std::plus<int>(), lambda);
    }

    //check result
    int expected[max_n];
    ::std::transform_exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected, 100, ::std::plus<int>(), lambda);

    EXPECT_EQ_N(expected, data1, max_n, "wrong effect from transform_exclusive_scan with init, sycl ranges");

#endif //_PSTL_USE_RANGES
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
