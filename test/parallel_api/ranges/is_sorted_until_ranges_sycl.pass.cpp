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

#include "support/pstl_test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    const int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const int idx = 5;
    data[idx] = 0;

    int res1 = -1, res2 = - 1;
    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view  = all_view(A);

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        res1 = is_sorted_until(exec, view);
        res2 = is_sorted_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), view, [](auto a, auto b) { return a < b; });
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'is_sorted_until' with sycl ranges");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'is_sorted_until' with comparator, sycl ranges");

<<<<<<< HEAD
#endif //_ENABLE_RANGES_TESTING
=======
#else
    TestUtils::skip();

#endif //_ONEDPL_USE_RANGES
>>>>>>> Added test skip invokes

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
