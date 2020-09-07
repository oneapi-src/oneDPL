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
    int data1[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int data2[max_n] = {0, 1, 2, -1, 4, 5, 6, 7, 8, 9};

    bool res1 = false;
    bool res2 = false;    
    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        cl::sycl::buffer<int> A(data1, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B(data2, cl::sycl::range<1>(max_n));

        auto exec = oneapi::dpl::execution::dpcpp_default;
        using Policy = decltype(oneapi::dpl::execution::dpcpp_default);

        res1 = is_sorted(exec, all_view(A));
        res2 = is_sorted(make_new_policy<new_kernel_name<Policy, 0>>(exec), all_view(B));
    }

    //check result
    EXPECT_TRUE(res1, "wrong effect from 'is_sorted' with sycl ranges (sorted)");
    EXPECT_TRUE(!res2, "wrong effect from 'is_sorted' with sycl ranges (unsorted)");
#endif //_PSTL_USE_RANGES

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
