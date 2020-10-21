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

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    auto res1 = -1, res2 = -1, res3 = -1;
    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));

        auto view = all_view<int, cl::sycl::access::mode::read>(A);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        res1 = reduce(exec, view);
        res2 = reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, 100);
        res3 = reduce(make_new_policy<new_kernel_name<Policy, 1>>(exec), view, 100, ::std::plus<int>());
    }

    //check result
    auto expected1 = ::std::accumulate(data, data + max_n, 0);
    auto expected2 = ::std::accumulate(data, data + max_n, 100);
    auto expected3 = expected2;

    EXPECT_TRUE(res1 == expected1, "wrong effect from reduce with sycl ranges");
    EXPECT_TRUE(res2 == expected2, "wrong effect from reduce with init, sycl ranges");
    EXPECT_TRUE(res3 == expected3, "wrong effect from reduce with init and binary operation, sycl ranges");
#endif //_PSTL_USE_RANGES
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
