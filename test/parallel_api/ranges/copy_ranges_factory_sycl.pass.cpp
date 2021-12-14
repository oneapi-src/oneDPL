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
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n];

    auto lambda1 = [](auto i) { return i * i; };

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(max_n);

        auto view = iota_view(0, max_n) | views::transform(lambda1);
        auto range_res = all_view<int, sycl::access::mode::write>(B);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec);
        auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        copy(exec1, view, C); //check passing a buffer for writing
        copy(exec2, C, range_res); //check passing a buffer for reading
    }

    //check result
    int expected[max_n];
    ::std::transform(data, data + max_n, expected, lambda1);

    EXPECT_EQ_N(expected, data2, max_n, "wrong effect from copy with factory and sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
