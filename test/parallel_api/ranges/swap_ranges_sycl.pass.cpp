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

#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    constexpr int max_n_2 = 5;
    int data1[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n]     = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int data3[max_n_2]   = {-1, -1, -1, -1, -1};
    int data4[max_n]     = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(data3, sycl::range<1>(max_n_2));
        sycl::buffer<int> D(data4, sycl::range<1>(max_n));
                          
        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);
        auto exec3 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 2>>(exec);
             
        swap_ranges(exec1, views::all(A), B);
        swap_ranges(exec2, B, C);
        swap_ranges(exec3, C, D);
    }

    //check result
    // data1  = { 9,  8,  7,  6,  5, 4, 3, 2, 1, 0};
    // data2  = {-1, -1, -1, -1, -1, 5, 6, 7, 8, 9};
    // data3  = { 0,  0,  0,  0,  0};
    // data4  = { 0,  1,  2,  3,  4, 0, 0, 0, 0, 0};

    auto expected1 = views::iota(0, max_n) | views::reverse;
    EXPECT_EQ_N(expected1.begin(), data1, max_n, "wrong result from swap");

    auto expected2_1 = views::fill(-1, max_n_2);
    auto expected2_2 = views::iota(max_n_2, max_n);
    EXPECT_EQ_N(expected2_1.begin(), data2, max_n_2, "wrong result from swap");
    EXPECT_EQ_N(expected2_2.begin(), data2 + max_n_2, max_n_2, "wrong result from swap");

    auto expected3 = views::fill(0, max_n_2);
    EXPECT_EQ_N(expected3.begin(), data3, max_n_2, "wrong result from swap");

    auto expected4_1 = views::iota(0, max_n_2);
    auto expected4_2 = expected3;
    EXPECT_EQ_N(expected4_1.begin(), data4, max_n_2, "wrong result from swap");
    EXPECT_EQ_N(expected4_2.begin(), data4 + max_n_2, max_n_2, "wrong result from swap");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
