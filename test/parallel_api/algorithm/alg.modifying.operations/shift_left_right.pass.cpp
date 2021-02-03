// -*- C++ -*-
//===-- shift_left.pass.cpp -----------------------------------------------===//
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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <list>
#include <iomanip>

#if  !defined(_PSTL_TEST_SHIFT_LEFT) && !defined(_PSTL_TEST_SHIFT_RIGHT)
#define _PSTL_TEST_SHIFT_LEFT
#define _PSTL_TEST_SHIFT_RIGHT
#endif

struct test_shift
{
    template <typename Policy, typename It, typename Size, typename Algo>
    oneapi::dpl::__internal::__enable_if_host_execution_policy<Policy, void>
    operator()(Policy&& exec, It first, Size m, It first_exp, Size n, Algo algo)
    {
        //run a test with host policy and host itertors
        It res = algo(::std::forward<Policy>(exec), first, ::std::next(first, m), n);
        //check result
        algo.check(res, first, m, first_exp, n);
    }

#if _ONEDPL_BACKEND_SYCL
    template <typename Policy, typename It, typename Size, typename Algo>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<Policy, void>
    operator()(Policy&& exec, It first, Size m, It first_exp, Size n, Algo algo)
    {
        //1.1 run a test with hetero policy and host itertors
        auto res = algo(::std::forward<Policy>(exec), first, first + m, n);
        //1.2 check result
        algo.check(res, first, m, first_exp, n);

        using _ValueType = typename ::std::iterator_traits<It>::value_type;

        //2.1 run a test with hetero policy and hetero itertors
        Size res_idx(0);
        {//scope for SYCL buffer lifetime
            sycl::buffer<_ValueType> buf(first, first + m);
            buf.set_final_data(first);
            buf.set_write_back(true);

            auto het_begin = oneapi::dpl::begin(buf);

            auto het_res = algo(::std::forward<Policy>(exec), het_begin, het_begin + m, n);
            res_idx = het_res - het_begin;
        }
        //2.2 check result
        algo.check(first + res_idx, first, m, first_exp, n);

#if _PSTL_SYCL_TEST_USM
        //3.1 run a test with hetero policy and USM pointers
        {
            // allocate USM memory
            auto queue = exec.queue();
            auto sycl_deleter = [queue](_ValueType* mem) { sycl::free(mem, queue.get_context()); };
            ::std::unique_ptr<_ValueType, decltype(sycl_deleter)> ptr(
                (_ValueType*)sycl::malloc_shared(sizeof(_ValueType)*m, queue.get_device(), queue.get_context()),
                sycl_deleter);

            //copying data to USM buffer
            ::std::copy_n(first, m, ptr.get());

            auto het_res = algo(::std::forward<Policy>(exec), ptr.get(), ptr.get() + m, n);
            res_idx = het_res - ptr.get();

            //3.2 check result
            algo.check(ptr.get() + res_idx, ptr.get(), m, first_exp, n);
        }
#endif
    }
#endif
};

template <typename T, typename Size, typename Algo>
void
test_shift_by_type(Size m, Size n, Algo algo)
{
    TestUtils::Sequence<T> orig(m, [](::std::size_t v) -> T { return T(v); }); //fill data
    TestUtils::Sequence<T> in(m, [](::std::size_t v) -> T { return T(v); }); //fill data

    TestUtils::invoke_on_all_policies<>()(test_shift(), in.begin(), m, orig.begin(), n, algo);
}

struct shift_left_algo
{
    template <typename Policy, typename It, typename Size>
    It operator()(Policy&& exec, It first, It last, Size n)
    {
        return oneapi::dpl::shift_left(::std::forward<Policy>(exec), first, last, n);
    }

    template <typename It, typename ItExp, typename Size>
    void
    check(It res, It first, Size m, ItExp first_exp, Size n)
    {
        //if (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first.
        //Otherwise, returns last.
        It __last = ::std::next(first, m);
        auto res_exp = (n > 0 && n < m ? ::std::next(first, m - n) : (n > 0 ? first : __last));

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_left");

        if(res != first && res != __last)
        {
            EXPECT_EQ_N(first, ::std::next(first_exp, + n), m - n, "wrong effect of shift_left");
            //restore unput data
            std::copy_n(first_exp, m, first);
        }
    }
};

struct shift_right_algo
{
    template <typename Policy, typename It, typename Size>
    typename ::std::enable_if<TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value
                            || TestUtils::is_same_iterator_category<It, ::std::random_access_iterator_tag>::value,
                            It>::type
    operator()(Policy&& exec, It first, It last, Size n)
    {
        return oneapi::dpl::shift_right(::std::forward<Policy>(exec), first, last, n);
    }
    //skip the test for non-bidirectional iterator (forward iterator, etc)
    template <typename Policy, typename It, typename Size>
    typename ::std::enable_if<!TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value
                            && !TestUtils::is_same_iterator_category<It, ::std::random_access_iterator_tag>::value,
                            It>::type
    operator()(Policy&& exec, It first, It last, Size n)
    {
        return first;
    }

    template <typename It, typename ItExp, typename Size>
    typename ::std::enable_if<TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value
                            || TestUtils::is_same_iterator_category<It, ::std::random_access_iterator_tag>::value,
                            void>::type
    check(It res, It first, Size m, ItExp first_exp, Size n)
    {
        //if (n > 0 && n < m), returns first + n. Otherwise, if n  > 0, returns last.
        //Otherwise, returns firts.
        It __last = ::std::next(first, m);
        auto res_exp = (n > 0 && n < m ? ::std::next(first, n) : (n > 0 ? __last : first));

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_right");

        if(res != first && res != __last)
        {
            EXPECT_EQ_N(::std::next(first, n), first_exp, m - n, "wrong effect of shift_right");
            //restore unput data
            std::copy_n(first_exp, m, first);
        }
    }
    //skip the check for non-bidirectional iterator (forward iterator, etc)
    template <typename It, typename ItExp, typename Size>
    typename ::std::enable_if<!TestUtils::is_same_iterator_category<It, ::std::bidirectional_iterator_tag>::value
                            && !TestUtils::is_same_iterator_category<It, ::std::random_access_iterator_tag>::value,
                            void>::type
    check(It res, It first, Size m, ItExp first_exp, Size n)
    {
    }
};

int
main()
{
    const ::std::size_t N = 100000;
    for (long m = 0; m < N; m = m < 16 ? m + 1 : long(3.1415 * m))
        for (long n = 0; n < N; n = n < 16 ? n + 1 : long(3.1415 * n))
    {
       //std::cout << "m: " << m << " n: " << n << std::endl;
#ifdef _PSTL_TEST_SHIFT_LEFT
       test_shift_by_type<int32_t>(m, n, shift_left_algo{}); 
#endif
#ifdef _PSTL_TEST_SHIFT_RIGHT
       test_shift_by_type<int32_t>(m, n, shift_right_algo{});
#endif
    }

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
