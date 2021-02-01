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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)

#if _ONEDPL_USE_RANGES
#include _PSTL_TEST_HEADER(ranges)
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ONEDPL_USE_RANGES
    const int max_n = 10;
    int data1[max_n] = { 0,  1, 2, 3, 4, 5, 6, 6, 6, 6};
    int data2[max_n] = {-2, -1, 2, 3, 3, 5, 6, 7, 8, 9};
    int data3[2 * max_n] = {};

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data1, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));
        sycl::buffer<int> C(data3, sycl::range<1>(max_n));

        auto exec = TestUtils::default_dpcpp_policy;
        using Policy = decltype(TestUtils::default_dpcpp_policy);

        merge(exec, all_view(A), all_view(B), all_view<int, sycl::access::mode::write>(C));
    }

    //check result
    bool res = ::std::is_sorted(data3, data3 + max_n);
    EXPECT_TRUE(res, "wrong effect from 'merge' with sycl ranges");
#endif //_ONEDPL_USE_RANGES

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
