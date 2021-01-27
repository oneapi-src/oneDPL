// -*- C++ -*-
//===-- async_sycl.pass.cpp -----------------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/async"
#include "oneapi/dpl/iterator"

#include <iostream>
#include <iomanip>

#include <CL/sycl.hpp>

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

int
main()
{
    const int n = 100;
    {
        sycl::queue q;
        sycl::buffer<int> x{n};
        sycl::buffer<int> y{n};

        auto my_policy = oneapi::dpl::execution::make_device_policy(q);
        auto res_1a = oneapi::dpl::experimental::copy_async(my_policy, oneapi::dpl::counting_iterator<int>(0),
                                                    oneapi::dpl::counting_iterator<int>(n), oneapi::dpl::begin(x)); // x = [0..n]
        auto res_1b = oneapi::dpl::experimental::fill_async(my_policy, oneapi::dpl::begin(y), oneapi::dpl::end(y), 7); // y = [7..7]

        auto res_2a = oneapi::dpl::experimental::for_each_async(
            my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x), [](auto& e) { ++e; }, res_1a); // x = [1..n]
        auto res_2b = oneapi::dpl::experimental::transform_async(
            my_policy, oneapi::dpl::begin(y), oneapi::dpl::end(y), oneapi::dpl::begin(y), [](const auto& e) { return e / 2; },
            res_1b); // y = [3..3]

        sycl::buffer<int> z{n}; //std::vector<int> z(n);
        auto res_3 =
            oneapi::dpl::experimental::transform_async(my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x), oneapi::dpl::begin(y), oneapi::dpl::begin(z),
                                               std::plus<int>(), res_2a, res_2b); // z = [4..n+3]

        auto alpha = oneapi::dpl::experimental::reduce_async(my_policy, oneapi::dpl::begin(x), oneapi::dpl::end(x), 0, std::plus<int>(),
                                                     res_2a).get(); // alpha = n*(n+1)/2

        auto beta = oneapi::dpl::experimental::transform_reduce_async(my_policy, oneapi::dpl::begin(z), oneapi::dpl::end(z), 0,
                                                                std::plus<int>(), [=](auto e) { return alpha * e; }).get();

        ASSERT_EQUAL(beta, (n * (n + 1) / 2) * ((n + 3) * (n + 4) / 2 - 6));
    }
    std::cout << "done" << std::endl;
    return 0;
}
