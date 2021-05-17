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

        auto my_policy = oneapi::dpl::execution::make_device_policy<class Copy>(oneapi::dpl::execution::dpcpp_default);
        auto input = oneapi::dpl::counting_iterator<int>(0);
        
        dpl::experimental::copy_async(my_policy, input, input+n, dpl::begin(x)).wait();
        const auto expected1 = ((n-1)*n)/2;
        const auto expected2 = expected1-n+1;

        auto my_policy1 = oneapi::dpl::execution::make_device_policy<class Scan1>(my_policy);
        auto alpha = dpl::experimental::transform_inclusive_scan_async(my_policy1, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), [](auto x) { return x * 10; });
        auto result1 = alpha.get().get_buffer().get_access<sycl::access::mode::read>()[n-1]; 
        EXPECT_TRUE(result1 == expected1 * 10, "wrong effect from async scan test (I) with sycl buffer");
        
        auto my_policy3 = oneapi::dpl::execution::make_device_policy<class Scan3>(my_policy);
        auto beta = dpl::experimental::transform_exclusive_scan_async(my_policy3, dpl::begin(x), dpl::end(x), dpl::begin(y), 0, std::plus<int>(), [](auto x) { return x * 10; });
        auto result2 = beta.get().get_buffer().get_access<sycl::access::mode::read>()[n-1];
        EXPECT_TRUE(result2 == expected2 * 10, "wrong effect from async scan test (II) with sycl buffer");

        auto my_policy4 = oneapi::dpl::execution::make_device_policy<class Scan4>(my_policy);
        auto gamma = dpl::experimental::inclusive_scan_async(my_policy4, dpl::begin(x), dpl::end(x), dpl::begin(y));
        auto result3 = gamma.get().get_buffer().get_access<sycl::access::mode::read>()[n-1]; 
        EXPECT_TRUE(result3 == expected1, "wrong effect from async scan test (III) with sycl buffer");

        auto my_policy7 = oneapi::dpl::execution::make_device_policy<class Scan7>(my_policy);
        auto delta = dpl::experimental::exclusive_scan_async(my_policy7, dpl::begin(x), dpl::end(x), dpl::begin(y), 0);
        auto result4 = delta.get().get_buffer().get_access<sycl::access::mode::read>()[n-1];
        EXPECT_TRUE(result4 == expected2, "wrong effect from async scan test (IV) with sycl buffer");

        oneapi::dpl::experimental::wait_for_all(alpha,beta,gamma,delta);   
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
