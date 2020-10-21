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
    int data1[max_n], data2[max_n];

    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B1(data1, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B2(data2, cl::sycl::range<1>(max_n));

        using namespace TestUtils;
        using namespace oneapi::dpl::experimental;

        auto view = ranges::all_view<int, cl::sycl::access::mode::read>(A);
        auto view_res1 = ranges::all_view<int, cl::sycl::access::mode::write>(B1);
        auto view_res2 = ranges::all_view<int, cl::sycl::access::mode::write>(B2);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        ranges::exclusive_scan(exec, view, view_res1, 100);
        ranges::exclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, view_res2, 100, ::std::plus<int>());
    }

    //check result
    int expected1[max_n], expected2[max_n], expected3[max_n];
    ::std::exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected1, 100);
    ::std::exclusive_scan(oneapi::dpl::execution::seq, data, data + max_n, expected2, 100, ::std::plus<int>());

    EXPECT_EQ_N(expected1, data1, max_n, "wrong effect from exclusive_scan with init, sycl ranges");
    EXPECT_EQ_N(expected2, data2, max_n, "wrong effect from exclusive_scan with init andbinary operation, sycl ranges");

#endif //_PSTL_USE_RANGES
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
