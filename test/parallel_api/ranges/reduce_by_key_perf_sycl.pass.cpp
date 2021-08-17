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
#include "oneapi/dpl/algorithm"
#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif
#include "support/utils.h"
#include <iostream>
#include <time.h>

#if TEST_DPCPP_BACKEND_PRESENT
template<typename BufA, typename BufB, typename BufC, typename BufD, typename Res>
auto do_work_2(BufA A, BufB B, BufC C, BufD D, Res& res)
{
    using namespace oneapi::dpl::experimental::ranges;

    auto sTime = std::chrono::high_resolution_clock::now();
    res = reduce_by_segment(TestUtils::default_dpcpp_policy, views::all_read(A), views::all_read(B), views::all_write(C), views::all_write(D));
    auto eTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(eTime - sTime).count();
}

template<typename BufA, typename BufB, typename BufC, typename BufD, typename Res>
auto do_work_4(BufA A, BufB B, BufC C, BufD D, Res& res)
{
    using namespace oneapi::dpl::experimental::ranges;

    auto sTime = std::chrono::high_resolution_clock::now();
    res = oneapi::dpl::reduce_by_segment(TestUtils::default_dpcpp_policy, oneapi::dpl::begin(A), oneapi::dpl::end(A),
        oneapi::dpl::begin(B), oneapi::dpl::begin(C), oneapi::dpl::begin(D)).first - oneapi::dpl::begin(B);
    auto eTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(eTime - sTime).count();
}
#endif

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    const int max_n = 1e7;
    const int m = 7;

    sycl::buffer<int> A(max_n); //keys
    sycl::buffer<int> B(max_n); //values
    sycl::buffer<int> C(max_n); //out keys
    sycl::buffer<int> D(max_n); //out values

    sycl::buffer<int> C1(max_n); //out keys
    sycl::buffer<int> D1(max_n); //out values

    auto exec = TestUtils::default_dpcpp_policy;
    using namespace oneapi::dpl::experimental::ranges;
    
    //generate keys
    auto gen_key = [max_n, m](auto i) { i ^= i%m << 13; i ^= i >> 17; i ^= i << 5; return i % m << 2; };
    copy(exec, views::iota(0, max_n) | views::transform(gen_key), views::all_write(A));
    //generate values
    copy(exec, views::iota(0, max_n), views::all_write(B));

    auto res = 0; 
    auto time = (do_work_2(A, B, C, D, res), do_work_2(A, B, C, D, res)); //double call to init RT and warm the cache
    ::std::cout << "do_work, reduce_by_segment, 2 patterns, time (ms): " << time << ::std::endl;

    // dump first 100 elements
#if 1
    ::std::cout << "the first 100 input keys: ";
    for(int i = 0; i < 100; ++i)
       ::std::cout << A.template get_access<sycl::access_mode::read>()[i] << " ";
    ::std::cout << ::std::endl;
#endif

    auto res1 = 0; 
    auto time2 = (do_work_4(A, B, C1, D1, res1), do_work_4(A, B, C1, D1, res1)); //double call to init RT and warm the cache
    ::std::cout << "do_work reduce_by_segment, 4 patterns, time (ms): " << time2 << ::std::endl;

    // dump the first 100 elements
#if 0
    ::std::cout << "the first 100 output keys: ";
    for(int i = 0; i < 100; ++i)
       ::std::cout << C1.template get_access<sycl::access_mode::read>()[i] << " ";
    ::std::cout << ::std::endl;
#endif

    EXPECT_TRUE(res1 == res, "wrong result from reduce_by_segment");
    EXPECT_EQ_N(views::host_all(C1).begin(), views::host_all(C).begin(), res1, "wrong keys from reduce_by_segment");
    EXPECT_EQ_N(views::host_all(D1).begin(), views::host_all(D).begin(), res1, "wrong values from reduce_by_segment");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
