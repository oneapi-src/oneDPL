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

    auto pred = [](auto i) { return i % 2 == 0; };

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(max_n);
    sycl::buffer<int> B(max_n);
    
    auto exec = TestUtils::default_dpcpp_policy;
    using Policy = decltype(exec);
    auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
    auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

    auto src = views::iota(0, max_n);

    auto res1 = copy_if(exec1, src, A, pred);
    auto res2 = copy_if(exec2, src, views::all_write(B), ::std::not_fn(pred));

    EXPECT_TRUE(res1 == 5, "wrong return result from copy_if with sycl buffer");
    EXPECT_TRUE(res2 == 5, "wrong return result from copy_if with sycl ranges");

    //check result
    int expected[max_n];

    ::std::copy_if(src.begin(), src.end(), expected, pred);
    EXPECT_EQ_N(views::host_all(A).begin(), expected, res1, "wrong effect from copy_if with sycl ranges");

    ::std::copy_if(src.begin(), src.end(), expected, ::std::not_fn(pred));
    EXPECT_EQ_N(views::host_all(B).begin(), expected, res2, "wrong effect from copy_if with sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
