// -*- C++ -*-
//===-- set.pass.cpp ------------------------------------------------------===//
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
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include <cmath>
#include <chrono>

#if  !defined(_PSTL_TEST_SET_UNION) && !defined(_PSTL_TEST_SET_DIFFERENCE) && !defined(_PSTL_TEST_SET_INTERSECTION) &&\
     !defined(_PSTL_TEST_SET_SYMMETRIC_DIFFERENCE)
#define _PSTL_TEST_SET_UNION
#define _PSTL_TEST_SET_DIFFERENCE
#define _PSTL_TEST_SET_INTERSECTION
#define _PSTL_TEST_SET_SYMMETRIC_DIFFERENCE
#endif

using namespace TestUtils;

template <typename T>
struct Num
{
    T val;

    Num() : val{} {}
    Num(const T& v) : val(v) {}

    //for "includes" checks
    template <typename T1>
    bool
    operator<(const Num<T1>& v1) const
    {
        return val < v1.val;
    }

    //The types Type1 and Type2 must be such that an object of type InputIt can be dereferenced and then implicitly converted to both of them
    template <typename T1>
    operator Num<T1>() const
    {
        return Num<T1>((T1)val);
    }

    friend bool
    operator==(const Num& v1, const Num& v2)
    {
        return v1.val == v2.val;
    }
};

template <typename Type>
struct test_set_union
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename ::std::iterator_traits<InputIterator1>::value_type;

        auto n1 = ::std::distance(first1, last1);
        auto n2 = ::std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = ::std::set_union(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_union(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_union");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_union effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_intersection
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename ::std::iterator_traits<InputIterator1>::value_type;

        auto n1 = ::std::distance(first1, last1);
        auto n2 = ::std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = ::std::set_intersection(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_intersection(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_intersection");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_intersection effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename ::std::iterator_traits<InputIterator1>::value_type;

        auto n1 = ::std::distance(first1, last1);
        auto n2 = ::std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = ::std::set_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename Type>
struct test_set_symmetric_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<!TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        using T1 = typename ::std::iterator_traits<InputIterator1>::value_type;

        auto n1 = ::std::distance(first1, last1);
        auto n2 = ::std::distance(first2, last2);
        auto n = n1 + n2;
        Sequence<T1> expect(n);
        Sequence<T1> out(n);

        auto expect_res = ::std::set_symmetric_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_symmetric_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_symmetric_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res),
                    "wrong set_symmetric_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    typename ::std::enable_if<TestUtils::isReverse<InputIterator1>::value, void>::type
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
    }
};

template <typename T1, typename T2, typename Compare>
void
test_set(Compare compare)
{

    const ::std::size_t n_max = 100000;

    // The rand()%(2*n+1) encourages generation of some duplicates.
    ::std::srand(4200);

    for (::std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        for (::std::size_t m = 0; m < n_max; m = m <= 16 ? m + 1 : size_t(2.71828 * m))
        {
            //prepare the input ranges
            Sequence<T1> in1(n, [n](::std::size_t k) { return rand() % (2 * k + 1); });
            Sequence<T2> in2(m, [m](::std::size_t k) { return (m % 2) * rand() + rand() % (k + 1); });

            ::std::sort(in1.begin(), in1.end(), compare);
            ::std::sort(in2.begin(), in2.end(), compare);

#ifdef _PSTL_TEST_SET_UNION
            invoke_on_all_policies<0>()(test_set_union<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);
#endif

#ifdef _PSTL_TEST_SET_INTERSECTION
            invoke_on_all_policies<1>()(test_set_intersection<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);
#endif
#ifdef _PSTL_TEST_SET_DIFFERENCE
            invoke_on_all_policies<2>()(test_set_difference<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(),
                                        compare);
#endif
#ifdef _PSTL_TEST_SET_SYMMETRIC_DIFFERENCE
            invoke_on_all_policies<3>()(test_set_symmetric_difference<T1>(), in1.begin(), in1.end(), in2.cbegin(),
                                        in2.cend(), compare);
#endif
        }
    }
}

template <typename T>
struct test_non_const_set_difference
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_difference(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(::std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_intersection
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_intersection(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(::std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_symmetric_difference
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_symmetric_difference(exec, input_iter, input_iter, input_iter, input_iter, out_iter,
                                 non_const(::std::less<T>()));
    }
};

template <typename T>
struct test_non_const_set_union
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        set_union(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(::std::less<T>()));
    }
};

int
main()
{
    using data_t =
#if !_PSTL_FPGA_DEVICE
        float64_t;
#else
        int32_t;
#endif

    test_set<data_t, data_t>(oneapi::dpl::__internal::__pstl_less());

#if !_PSTL_BACKEND_SYCL
    test_set<Num<int64_t>, Num<int32_t>>([](const Num<int64_t>& x, const Num<int32_t>& y) { return x < y; });

    test_set<MemoryChecker, MemoryChecker>([](const MemoryChecker& val1, const MemoryChecker& val2) -> bool {
        return val1.value() < val2.value();
    });
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from set algorithms: number of ctor and dtor calls is not equal");
#endif

#ifdef _PSTL_TEST_SET_DIFFERENCE
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_difference<int32_t>>());
#endif
#ifdef _PSTL_TEST_SET_INTERSECTION
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_intersection<int32_t>>());
#endif
#ifdef _PSTL_TEST_SET_SYMMETRIC_DIFFERENCE
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_symmetric_difference<int32_t>>());
#endif
#ifdef _PSTL_TEST_SET_UNION
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_set_union<int32_t>>());
#endif

    ::std::cout << done() << ::std::endl;

    return 0;
}
