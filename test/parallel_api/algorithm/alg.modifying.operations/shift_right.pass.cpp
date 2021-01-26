// -*- C++ -*-
//===-- shift_right.pass.cpp -----------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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
struct test_shift_right
{
    template <typename Policy, typename It, typename Size>
    typename ::std::enable_if<TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, It first, Size m, It first_exp, Size n)
    {
	    auto res = oneapi::dpl::shift_right(exec, first, std::next(first, m), n);

        //if (n > 0 && n < m), returns first + n. Otherwise, if n  > 0, returns last.
        //Otherwise, returns first;
        auto res_exp = (n > 0 && n < m ? std::next(first, n) : (n > 0 ? std::next(first, m) : first));

#if 0
        std::cout << "m: " << m << " n: " << n << std::endl;
        std::cout << "res: " << res - first << std::endl;
        std::cout << "res_exp: " << res_exp - first << std::endl;
#endif

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_right");

        if(n > m || n < 0) //should be no effect in this case
	    n = 0;

        EXPECT_EQ_N(std::next(first, + n), first_exp, m - n, "wrong effect of shift_right");

        //restore unput data
       std::copy(first_exp, std::next(first_exp, m), first);
    }

    template <typename Policy, typename It, typename Size>
    typename ::std::enable_if<!TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, It first, Size m, It first_exp, Size n)
    {
    }    
};

template <typename T, typename Size>
void
test_shift_right_by_type(Size m, Size n)
{
    TestUtils::Sequence<T> orig(m, [](::std::size_t v) -> T { return T(v); }); //fill data
    TestUtils::Sequence<T> in(m, [](::std::size_t v) -> T { return T(v); }); //fill data

    TestUtils::invoke_on_all_policies()(test_shift_right<T>(), in.begin(), m, orig.begin(), n);
}

int
main()
{
    const ::std::size_t N = 100000;
    for (long m = 0; m < N; m = m < 16 ? m + 1 : long(3.1415 * m))
        for (long n = 0; n < N; n = n < 16 ? n + 1 : long(3.1415 * n))
    {
        test_shift_right_by_type<int32_t>(m, n);
    }

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}