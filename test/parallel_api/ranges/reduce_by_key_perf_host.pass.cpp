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

#include <iostream>
#include <time.h>
#include <vector>
#include <oneapi/dpl/execution>
#include "oneapi/dpl/algorithm"
#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif
#include "support/utils.h"

template<typename BufA, typename BufB, typename BufC, typename BufD, typename Res>
auto do_work_2(BufA& A, BufB& B, BufC& C, BufD& D, Res& res)
{
    using namespace oneapi::dpl::experimental::ranges;

    auto sTime = std::chrono::high_resolution_clock::now();
    res = reduce_by_segment(::std::execution::par_unseq, nano::views::all(A), nano::views::all(B), nano::views::all(C), nano::views::all(D));
    auto eTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(eTime - sTime).count();
}

template<typename BufA, typename BufB, typename BufC, typename BufD, typename Res>
auto do_work_4(BufA& A, BufB& B, BufC& C, BufD& D, Res& res)
{
    using namespace oneapi::dpl::experimental::ranges;

    auto sTime = std::chrono::high_resolution_clock::now();
    res = oneapi::dpl::reduce_by_segment(::std::execution::par_unseq, ::std::begin(A), ::std::end(A),
        ::std::begin(B), ::std::begin(C), ::std::begin(D)).first - ::std::begin(B);
    auto eTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(eTime - sTime).count();
}


int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    const int max_n = 1e7;
    const int m = 7;

    ::std::vector<int> A(max_n); //keys
    ::std::vector<int> B(max_n); //values
    ::std::vector<int> C(max_n); //out keys
    ::std::vector<int> D(max_n); //out values

    ::std::vector<int> C1(max_n); //out keys
    ::std::vector<int> D1(max_n); //out values

    using namespace oneapi::dpl::experimental::ranges;
    
    //generate keys
    auto exec = ::std::execution::par_unseq;
    auto gen_key = [max_n, m](auto i) { i ^= i%m << 13; i ^= i >> 17; i ^= i << 5; return i % m << 2; };
    auto __iota = views::iota(0, max_n);
    auto __v = __iota | views::transform(gen_key);
    ::std::copy(__v.begin(), __v.end(), A.begin());
    //generate values
    ::std::copy(__iota.begin(), __iota.end(), B.begin());

    auto res = 0; 
    auto time = (do_work_2(A, B, C, D, res), do_work_2(A, B, C, D, res)); //double call to init RT and warm the cache
    ::std::cout << "do_work, reduce_by_segment, 2 patterns, time (ms): " << time << ::std::endl;
//    ::std::cout << res << ::std::endl;

    // dump first 100 elements
#if 1
    ::std::cout << "the first 100 input keys: ";
    for(int i = 0; i < 100; ++i)
       ::std::cout << A[i] << " ";
    ::std::cout << ::std::endl;
#endif

    auto res1 = 0; 
    auto time2 = (do_work_4(A, B, C1, D1, res1), do_work_4(A, B, C1, D1, res1)); //double call to init RT and warm the cache
    ::std::cout << "do_work reduce_by_segment, 4 patterns, time (ms): " << time2 << ::std::endl;
//    ::std::cout << res1 << ::std::endl;

    // dump the first 100 elements
#if 0
    ::std::cout << "the first 100 output keys: ";
    for(int i = 0; i < 100; ++i)
       ::std::cout << C1[i] << " ";
    ::std::cout << ::std::endl;
#endif

//    EXPECT_TRUE(res1 == res, "wrong result from reduce_by_segment");
    EXPECT_EQ_N(C1.begin(), C.begin(), res1, "wrong keys from reduce_by_segment");
    EXPECT_EQ_N(D1.begin(), D.begin(), res1, "wrong values from reduce_by_segment");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
