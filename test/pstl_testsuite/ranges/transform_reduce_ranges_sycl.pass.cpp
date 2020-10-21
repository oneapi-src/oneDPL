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
#if _PSTL_USE_RANGES
#include _PSTL_TEST_HEADER(ranges)
#endif

int32_t
main()
{
#if _PSTL_USE_RANGES
    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto lambda1 = [](auto val) { return val = val * val; };

    auto res1 = -1, res2 = -1, res3 = -1;
    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));

        using namespace TestUtils;

        auto view = oneapi::dpl::experimental::ranges::all_view<int, cl::sycl::access::mode::read>(A);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        res1 = oneapi::dpl::experimental::ranges::transform_reduce(exec, view, view, 0);
        res2 = oneapi::dpl::experimental::ranges::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, view, 0, ::std::plus<int>(), ::std::multiplies<int>());
        res3 = oneapi::dpl::experimental::ranges::transform_reduce(make_new_policy<new_kernel_name<Policy, 1>>(exec), view, 0, ::std::plus<int>(), lambda1);
    }

    //check result
    auto expected1 = ::std::inner_product(data, data + max_n, data, 0);
    auto expected2 = ::std::inner_product(data, data + max_n, data, 0, ::std::plus<int>(), ::std::multiplies<int>());

    auto data_view = nano::views::all(data) | oneapi::dpl::experimental::ranges::views::transform(lambda1);
    auto expected3 = ::std::accumulate(data_view.begin(), data_view.end(), 0);

    EXPECT_TRUE(res1 == expected1, "wrong effect from transform_reduce1 with sycl ranges");
    EXPECT_TRUE(res2 == expected2, "wrong effect from transform_reduce2 with sycl ranges");
    EXPECT_TRUE(res3 == expected3, "wrong effect from transform_reduce3 with sycl ranges");
#endif //_PSTL_USE_RANGES
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
