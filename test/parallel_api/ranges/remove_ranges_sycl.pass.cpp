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

#include <vector>
#include <iostream>
#include <iterator>

template<typename T>
class print_t;

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    using T = int;
    ::std::vector<T> data = {2, 5, 2, 4, 4, 0, 6, -7, 7, 3};
    const int max_n = data.size();

    auto lambda = [](T val){ return val % 2 == 0; };
    T val = 2;

    ::std::vector<T> in1(data);
    ::std::vector<T> in2(data);

    using namespace TestUtils;
    using namespace oneapi::dpl::experimental::ranges;

    ::std::vector<T>::difference_type in1_end_n;
    ::std::vector<T>::difference_type in2_end_n;
    {
        sycl::buffer<T> A(in1.data(), sycl::range<1>(max_n));
        sycl::buffer<T> B(in2.data(), sycl::range<1>(max_n));

        using Policy = decltype(TestUtils::default_dpcpp_policy);
        auto exec = TestUtils::default_dpcpp_policy;
        auto exec1 = make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
        auto exec2 = make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

        in1_end_n = remove(exec1, all_view<T, sycl::access::mode::read_write>(A), val);
        in2_end_n = remove_if(exec2, all_view<T, sycl::access::mode::read_write>(B), lambda);
    }

    //check result
    ::std::vector<T> exp2(data);
    ::std::vector<T> exp1(data);

    auto exp1_end = ::std::remove(exp1.begin(), exp1.end(), val);
    auto exp2_end = ::std::remove_if(exp2.begin(), exp2.end(), lambda);

    EXPECT_TRUE(::std::distance(exp1.begin(), exp1_end) == in1_end_n, "wrong effect from remove with sycl ranges");
    EXPECT_TRUE(::std::distance(exp2.begin(), exp2_end) == in2_end_n, "wrong effect from remove_if with sycl ranges");

    EXPECT_EQ_N(exp1.begin(), in1.begin(), in1_end_n, "wrong effect from remove with sycl ranges");
    EXPECT_EQ_N(exp2.begin(), in2.begin(), in2_end_n, "wrong effect from remove_if with sycl ranges");
#endif //_ENABLE_RANGES_TESTING

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}

