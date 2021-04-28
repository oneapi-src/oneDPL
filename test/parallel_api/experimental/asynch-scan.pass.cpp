// -*- C++ -*-
//===-- async-scan.pass.cpp ----------------------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#   include "oneapi/dpl/async"
#   include <CL/sycl.hpp>
#endif

#include <iostream>
#include <iomanip>
#include <numeric>

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (ASYNC): fail (" << X << "," << Y << ")" << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
void
test_with_buffers()
{
    const int n = 100;
    {
        sycl::buffer<int> x{n};
        sycl::buffer<int> y{n};

        auto my_policy = oneapi::dpl::execution::make_device_policy<class Scan>(oneapi::dpl::execution::dpcpp_default);

        // ()
        auto my_policy1 = oneapi::dpl::execution::make_device_policy<class Scan1>(my_policy);
        auto res1a = dpl::experimental::transform_inclusive_scan_async(my_policy1, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), [](auto x) { return x * 10; }).get();
        auto my_policy2 = oneapi::dpl::execution::make_device_policy<class Scan2>(my_policy);
        auto res1b = dpl::experimental::transform_inclusive_scan_async(my_policy2, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), [](auto x) { return x * 10; }, 0).get();

        // ()
        auto my_policy3 = oneapi::dpl::execution::make_device_policy<class Scan3>(my_policy);
        auto res2 = dpl::experimental::transform_exclusive_scan_async(my_policy3, dpl::begin(x), dpl::end(x), dpl::begin(y), 0, std::plus<int>(), [](auto x) { return x * 10; });

        // ()
        auto my_policy4 = oneapi::dpl::execution::make_device_policy<class Scan4>(my_policy);
        auto res3a = dpl::experimental::inclusive_scan_async(my_policy4, dpl::begin(x), dpl::end(x), dpl::begin(y));
        auto my_policy5 = oneapi::dpl::execution::make_device_policy<class Scan5>(my_policy);
        auto res3b = dpl::experimental::inclusive_scan_async(my_policy5, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>());
        auto my_policy6 = oneapi::dpl::execution::make_device_policy<class Scan6>(my_policy);
        auto res3c = dpl::experimental::inclusive_scan_async(my_policy6, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), 0);

        // ()
        auto my_policy7 = oneapi::dpl::execution::make_device_policy<class Scan7>(my_policy);
        auto res4a = dpl::experimental::exclusive_scan_async(my_policy7, dpl::begin(x), dpl::end(x), dpl::begin(y), 0);
        auto my_policy8 = oneapi::dpl::execution::make_device_policy<class Scan8>(my_policy);
        auto res4b = dpl::experimental::exclusive_scan_async(my_policy8, dpl::begin(x), dpl::end(x), dpl::begin(y), 0, std::plus<int>());
    
        // TODO: Add check!    
    }
}
#endif

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
#endif
    std::cout << "done" << std::endl;
    return 0;
}
