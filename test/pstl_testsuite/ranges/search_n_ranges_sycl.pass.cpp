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
    const int count = 10;
    int data[count] = {0, 1, 2, 3, 4, 4, 4, 7, 8, 9};

    const int n_val = 3;
    const int idx = 4;
    const int val = data[idx];
    int res = -1;

    using namespace oneapi::dpl::experimental::ranges;
    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(count));

        auto view_a = all_view(A);
        res = search_n(oneapi::dpl::execution::dpcpp_default, view_a, n_val, val, [](auto a, auto b) { return a == b; });
    }

    //check result
    EXPECT_TRUE(res == idx, "wrong effect from 'search_n' with sycl ranges");

#endif //_PSTL_USE_RANGES

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
