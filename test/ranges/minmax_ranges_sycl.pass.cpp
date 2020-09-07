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
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));

        auto view = all_view(A);

        auto exec = oneapi::dpl::execution::dpcpp_default;
        using Policy = decltype(oneapi::dpl::execution::dpcpp_default);

        //min element
        res1 = min_element(exec, view);
        res2 = min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, ::std::less<int>());
        res3 = min_element(make_new_policy<new_kernel_name<Policy, 1>>(exec), view | views::take(1));

        //max_element
        res4 = max_element(make_new_policy<new_kernel_name<Policy, 2>>(exec), view);
        res5 = max_element(make_new_policy<new_kernel_name<Policy, 3>>(exec), view, ::std::less<int>());
 
        res_minmax1 = minmax_element(make_new_policy<new_kernel_name<Policy, 4>>(exec), view);
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

#endif //_PSTL_USE_RANGES

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
