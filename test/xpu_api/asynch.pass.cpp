// -*- C++ -*-
//===-- async.pass.cpp ----------------------------------------------------===//
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
#include "oneapi/dpl/async"
#include "oneapi/dpl/iterator"

#include "support/pstl_test_config.h"

#include <iostream>
#include <iomanip>

#if TEST_SYCL_PRESENT
#    include <CL/sycl.hpp>
#endif

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
        sycl::queue q;
        sycl::buffer<int> x{n};
        sycl::buffer<int> y{n};

        auto my_policy = oneapi::dpl::execution::make_device_policy(q);
        auto res_1a = oneapi::dpl::experimental::copy_async(my_policy, oneapi::dpl::counting_iterator<int>(0),
                                                            oneapi::dpl::counting_iterator<int>(n),
                                                            oneapi::dpl::begin(x)); // x = [0..n]
        auto res_1b = oneapi::dpl::experimental::fill_async(my_policy, oneapi::dpl::begin(y), oneapi::dpl::end(y),
                                                            7); // y = [7..7]

        auto res_2a = oneapi::dpl::experimental::for_each_async(
            my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x), [](int& e) { ++e; }, res_1a); // x = [1..n]
        auto res_2b = oneapi::dpl::experimental::transform_async(
            my_policy, oneapi::dpl::begin(y), oneapi::dpl::end(y), oneapi::dpl::begin(y),
            [](const int& e) { return e / 2; },
            res_1b); // y = [3..3]

        sycl::buffer<int> z{n}; //std::vector<int> z(n);
        auto res_3 = oneapi::dpl::experimental::transform_async(my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x),
                                                                oneapi::dpl::begin(y), oneapi::dpl::begin(z),
                                                                std::plus<int>(), res_2a, res_2b); // z = [4..n+3]

        auto alpha = oneapi::dpl::experimental::reduce_async(my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x), 0,
                                                             std::plus<int>(),
                                                             res_2a)
                         .get(); // alpha = n*(n+1)/2

        auto beta =
            oneapi::dpl::experimental::transform_reduce_async(my_policy, oneapi::dpl::begin(z), oneapi::dpl::end(z), 0,
                                                              std::plus<int>(), [=](int e) { return alpha * e; })
                .get();

        ASSERT_EQUAL(beta, (n * (n + 1) / 2) * ((n + 3) * (n + 4) / 2 - 6));
    }
}

void
test_with_usm()
{
    cl::sycl::queue q;
    const int n = 13;

    // ASYNC TEST USING USM //

    {
        // Allocate space for data using USM.
        uint64_t* data1 =
            static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* data2 =
            static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));

        //T data1[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
        //T data2[n1] = { 2, 3, 4, 5, 2, 2, 4, 4, 2, 2, 4, 4, 0 };

        // Initialize data
        for (int i = 0; i != n - 1; ++i)
        {
            data1[i] = i % 4 + 1;
            data2[i] = i % 4 + 2;
            if (i > 3)
            {
                ++i;
                data1[i] = data1[i - 1];
                data2[i] = data2[i - 1];
            }
        }
        data1[n - 1] = 0;
        data2[n - 1] = 0;

        // call first algorithm
        auto new_policy = oneapi::dpl::execution::make_device_policy<class async1>(q);
        auto fut1 = oneapi::dpl::experimental::reduce_async(new_policy, data1, data1 + n);
        auto res1 = fut1.get();

        // check values
        ASSERT_EQUAL(res1, 26);

        // call second algorithm
        auto new_policy2 = oneapi::dpl::execution::make_device_policy<class async2>(q);
        auto res2 = oneapi::dpl::experimental::transform_reduce_async(
                        new_policy2, data2, data2 + n, data1, 0, std::plus<uint64_t>(), std::multiplies<uint64_t>())
                        .get();

        // check values
        ASSERT_EQUAL(res2, 96);
    }
}
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
    test_with_usm();
#endif
    std::cout << "done" << std::endl;
    return 0;
}
