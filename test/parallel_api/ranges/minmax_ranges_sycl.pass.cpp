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
    const int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const int idx_val = 5;
    const int val = -1;
    data[idx_val] = val;
    const int idx_max = max_n - 1;

    int res1 = -1, res2 = - 1, res3 = -1, res4 = -1, res5 = -1;
    ::std::pair<int, int> res_minmax1(-1, -1);
    ::std::pair<int, int> res_minmax2(-1, -1);
    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = all_view(A);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        //min element
        res1 = min_element(exec, A);
        res2 = min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, ::std::less<int>());
        res3 = min_element(make_new_policy<new_kernel_name<Policy, 1>>(exec), view | views::take(1));

        //max_element
        res4 = max_element(make_new_policy<new_kernel_name<Policy, 2>>(exec), A);
        res5 = max_element(make_new_policy<new_kernel_name<Policy, 3>>(exec), view, ::std::less<int>());

        res_minmax1 = minmax_element(make_new_policy<new_kernel_name<Policy, 4>>(exec), A);
        res_minmax2 = minmax_element(make_new_policy<new_kernel_name<Policy, 5>>(exec), view, ::std::less<int>());
    }

    //check result
    EXPECT_TRUE(res1 == idx_val, "wrong effect from 'min_element', sycl ranges");
    EXPECT_TRUE(res2 == idx_val, "wrong effect from 'min_element' with predicate,  sycl ranges");
    EXPECT_TRUE(res3 == 0, "wrong effect from 'min_element' with trivial sycl ranges");

    EXPECT_TRUE(res4 == idx_max, "wrong effect from 'max_element', sycl ranges");
    EXPECT_TRUE(res5 == idx_max, "wrong effect from 'max_element' with predicate,  sycl ranges");

    EXPECT_TRUE(res_minmax1.first == idx_val && res_minmax1.second == idx_max, "wrong effect from 'minmax_element', sycl ranges");
    EXPECT_TRUE(res_minmax2.first == idx_val && res_minmax2.second == idx_max, "wrong effect from 'minmax_element' with predicate, sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
