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

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data1[max_n]     = {-1, 1, -1, 3, 4, 5, 6, -1, 8, 9};
    int data2[max_n]     = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

    auto lambda = [](auto i) { return i % 2 == 0; };

    bool res1 = 0, res2 = 0;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        auto exec1 = TestUtils::default_dpcpp_policy;
        using Policy = decltype(exec1);
        auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec1);
                                       
        res1 = any_of(exec1, views::all(A), lambda);
        res2 = all_of(exec2, B, lambda);
    }

    EXPECT_TRUE(res1, "wrong result from any_of with sycl ranges");
    EXPECT_TRUE(res2, "wrong result from all_of with sycl ranges");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
