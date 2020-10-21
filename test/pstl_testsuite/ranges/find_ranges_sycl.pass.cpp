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

    int res1 = -1, res2 = - 1, res3 = -1;
    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));

        auto view = all_view(A);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        res1 = find(exec, view, val);
        res2 = find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, [val](auto a) { return a == val;});
        res3 = find_if_not(make_new_policy<new_kernel_name<Policy, 1>>(exec), view, [val](auto a) { return a >= 0;});
    }

    //check result
    EXPECT_TRUE(res1 == idx_val, "wrong effect from 'find' with sycl ranges");
    EXPECT_TRUE(res2 == idx_val, "wrong effect from 'find_if' with sycl ranges");
    EXPECT_TRUE(res3 == idx_val, "wrong effect from 'find_if_not' with sycl ranges");

#endif //_PSTL_USE_RANGES

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
