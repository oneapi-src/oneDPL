// -*- C++ -*-
//===-- shift_left.pass.cpp -----------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation
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
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <list>
#include <iomanip>

template <typename T>
struct test_shift_left
{
    template <typename Policy, typename It, typename Size>
    void
    operator()(Policy&& exec, It first, Size m, It first_exp, Size n)
    {
#if 0
        std::cout << "origin: ";
        for(auto i = 0; i < m; ++i)
            std::cout << *(first + i) << " ";
        std::cout << std::endl;
#endif

	    auto res = oneapi::dpl::shift_left(exec, first, std::next(first, m), n);

#if 0
        std::cout << "shifted: ";
        for(auto i = 0; i < m-n; ++i)
            std::cout << *(first + i) << " ";
        std::cout << std::endl;
#endif

        //if (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first.
        //Otherwise, returns first + m (or last).
        auto res_exp = (n > 0 && n < m ? std::next(first, m - n) : (n > 0 ? first : std::next(first, m)));

#if 0
        std::cout << "m: " << m << " n: " << n << std::endl;
        std::cout << "res: " << res - first << std::endl;
        std::cout << "res_exp: " << res_exp - first << std::endl;
#endif

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_left");

        if(n > m || n < 0) //should be no effect in this case
	    n = 0;

        EXPECT_EQ_N(first, std::next(first_exp, + n), m - n, "wrong effect of shift_left");

        //restore unput data
       std::copy_n(first_exp, m, first);
    }
};

template <typename T, typename Size>
void
test_shift_left_by_type(Size m, Size n)
{
    TestUtils::Sequence<T> orig(m, [](::std::size_t v) -> T { return T(v); }); //fill data
    TestUtils::Sequence<T> in(m, [](::std::size_t v) -> T { return T(v); }); //fill data

    TestUtils::invoke_on_all_host_policies()(test_shift_left<T>(), in.begin(), m, orig.begin(), n);
}

int
main()
{
    const ::std::size_t N = 10000;
    for (long m = 0; m < N; m = m < 16 ? m + 1 : long(3.1415 * m))
        for (long n = 0; n < N; n = n < 16 ? n + 1 : long(3.1415 * n))
    {
        test_shift_left_by_type<int32_t>(m, n);
    }

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}