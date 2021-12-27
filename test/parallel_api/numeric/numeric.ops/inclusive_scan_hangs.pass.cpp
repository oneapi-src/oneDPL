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

//#define _GLIBCXX_USE_TBB_PAR_BACKEND 0 // libstdc++10

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iostream>
#include <vector>

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    constexpr int kItemsCount = 10;

    std::vector<int> v{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    sycl::queue syclQue(sycl::gpu_selector{});

    TestUtils::usm_data_transfer<sycl::usm::alloc::device> dt_helper(syclQue, v.begin(), v.end());
    int* dev_v = dt_helper.get_data();

    auto policy = oneapi::dpl::execution::make_device_policy(syclQue);
    oneapi::dpl::inclusive_scan(policy, dev_v, dev_v + kItemsCount, dev_v); //, oneapi::dpl::maximum<int>() );

    std::vector<int> h(kItemsCount);
    dt_helper.retrieve_data(h.begin());

    for (auto elem : h)
    {
        std::cout << elem << " ";
    }
    std::cout << "\n";
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
