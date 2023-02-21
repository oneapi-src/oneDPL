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

#include "sycl_iterator.pass.h"

#if TEST_DPCPP_BACKEND_PRESENT

DEFINE_TEST(test_uninitialized_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                                  value);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2),
                                 value),
                    "wrong effect from uninitialized_fill");
    }
};

DEFINE_TEST(test_uninitialized_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from uninitialized_fill_n");
    }
};

DEFINE_TEST(test_uninitialized_default_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                             first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), exp_value),
                    "wrong effect from uninitialized_default_construct");
    }
};

DEFINE_TEST(test_uninitialized_default_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, exp_value),
                    "wrong effect from uninitialized_default_construct_n");
    }
};

DEFINE_TEST(test_uninitialized_value_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                           first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), T1{}),
                    "wrong effect from uninitialized_value_construct");
    }
};

DEFINE_TEST(test_uninitialized_value_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, T1{}),
                    "wrong effect from uninitialized_value_construct_n");
    }
};

DEFINE_TEST(test_adjacent_find)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_find)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::equal_to<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check with no adjacent equal elements
        Iterator result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #1 no elements)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                    << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with the last adjacent element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = *(host_keys.get() + n - 2);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 2 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #2 the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with an adjacent element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + max_dis / 2) = *(host_keys.get() + max_dis / 2 - 1);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? it - 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #3 middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with an adjacent element (no predicate)
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #4 middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with the first adjacent element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = *host_keys.get();
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? first : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #5 the first element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_is_sorted_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted_until)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        Iterator result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from is_sorted_until (Test #1 sorted sequence)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #2 unsorted sequence - the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + /*max_idx*/ max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected = it;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #3 unsorted sequence - the middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element (no predicate)
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(
            result == expected,
            "wrong effect from is_sorted_until (Test #4 unsorted sequence - the middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the first element
        if (n > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected = n > 1 ? first + 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #5 unsorted sequence - the first element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1,
                                       [n](T1 x) { return x == n - 1; });
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from find_if_0");
            res0 = ::std::find(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, T1(1));
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from find_0");
        }
        // find_if
        auto res1 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1,
                                   [n](T1 x) { return x == n - 1; });
        wait_and_throw(exec);

        EXPECT_TRUE((res1 - first1) == n - 1, "wrong effect from find_if_1");

        auto res2 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1,
                                   [](T1 x) { return x == -1; });
        wait_and_throw(exec);

        EXPECT_TRUE(res2 == last1, "wrong effect from find_if_2");

        auto res3 = ::std::find_if(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1,
                                   [](T1 x) { return x % 2 == 0; });
        wait_and_throw(exec);

        EXPECT_TRUE(res3 == first1, "wrong effect from find_if_3");

        //find
        auto res4 = ::std::find(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, T1(-1));
        wait_and_throw(exec);

        EXPECT_TRUE(res4 == last1, "wrong effect from find");
    }
};

DEFINE_TEST(test_search_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_search_n)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        ::std::iota(host_keys.get(), host_keys.get() + n, T(5));

        // Search for sequence at the end
        {
            auto start = (n > 3) ? (n / 3 * 2) : (n - 1);

            ::std::fill(host_keys.get() + start, host_keys.get() + n, T(11));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, n - start, T(11));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_1");
        }
        // Search for sequence in the middle
        {
            auto start = (n > 3) ? (n / 3) : (n - 1);
            auto end = (n > 3) ? (n / 3 * 2) : n;

            ::std::fill(host_keys.get() + start, host_keys.get() + end, T(22));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, end - start, T(22));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_20");

            // Search for sequence of lesser size
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                ::std::max(end - start - 1, (size_t)1), T(22));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_21");
        }
        // Search for sequence at the beginning
        {
            auto end = n / 3;

            ::std::fill(host_keys.get(), host_keys.get() + end, T(33));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last, end, T(33));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_3");
        }
        // Search for sequence that covers the whole range
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T(44));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, n, T(44));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_4");
        }
        // Search for sequence which is not there
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 5>>(exec), first, last, 2, T(55));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_50");

            // Sequence is there but of lesser size(see search_n_3)
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 6>>(exec), first, last, (n / 3 + 1), T(33));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_51");
        }

        // empty sequence case
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 7>>(exec), first, first, 1, T(5 + n - 1));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_6");
        }
        // 2 distinct sequences, must find the first one
        if (n > 10)
        {
            auto start1 = n / 6;
            auto end1 = n / 3;

            auto start2 = (2 * n) / 3;
            auto end2 = (5 * n) / 6;

            ::std::fill(host_keys.get() + start1, host_keys.get() + end1, T(66));
            ::std::fill(host_keys.get() + start2, host_keys.get() + end2, T(66));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 8>>(exec), first, last,
                                     ::std::min(end1 - start1, end2 - start2), T(66));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start1, "wrong effect from search_7");
        }

        if (n == 10)
        {
            auto seq_len = 3;

            // Should fail when searching for sequence which is placed before our first iterator.
            ::std::fill(host_keys.get(), host_keys.get() + seq_len, T(77));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 9>>(exec), first + 1, last, seq_len, T(77));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_8");
        }
    }
};

DEFINE_TEST(test_remove)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;
        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = ::std::remove(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, T1(222 + pos));
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last - 1, "wrong result from remove");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove");
        }
    }
};

DEFINE_TEST(test_remove_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove_if)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = ::std::remove_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                   [=](T1 x) { return x == T1(222 + pos); });
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last - 1, "wrong result from remove_if");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove_if");
        }
    }
};

DEFINE_TEST(test_unique)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&index](IteratorValueType& value) { value = (index++ + 4) / 4; });
        host_keys.update_data();

        // invoke
        auto f = [](IteratorValueType a, IteratorValueType b) { return a == b; };
        auto result_last = ::std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, f);
        wait_and_throw(exec);

        auto result_size = result_last - first;

        std::int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
#if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "buffer size: got " << result_last - first << ", expected " << expected_size << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first1 + i) != i + 1)
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "got: " << *(host_first1 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique");
        }
    }
};

DEFINE_TEST(test_partition_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;
        using Iterator3ValueType = typename ::std::iterator_traits<Iterator3>::value_type;
        auto f = [](Iterator1ValueType value) { return (value % 3 == 0) && (value % 2 == 0); };

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, Iterator1ValueType{0});
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator2ValueType{-1});
        ::std::fill(host_res.get(),   host_res.get() + n, Iterator3ValueType{-2});
        update_data(host_keys, host_vals, host_res);

        // invoke
        auto res =
            ::std::partition_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first3, f);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals, host_res);

        // init for expected
        ::std::vector<Iterator2ValueType> exp_true(n, -1);
        ::std::vector<Iterator3ValueType> exp_false(n, -2);
        auto exp_true_first = exp_true.begin();
        auto exp_false_first = exp_false.begin();

        // invoke for expected
        auto exp = ::std::partition_copy(host_keys.get(), host_keys.get() + n, exp_true_first, exp_false_first, f);

        // check
        bool is_correct = (exp.first - exp_true_first) == (res.first - first2) &&
                          (exp.second - exp_false_first) == (res.second - first3);
#if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "N =" << n << ::std::endl
                      << "buffer size: got {" << res.first - first2 << "," << res.second - first3 << "}, expected {"
                      << exp.first - exp_true_first << "," << exp.second - exp_false_first << "}" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        for (int i = 0; i < ::std::min(exp.first - exp_true_first, res.first - first2) && is_correct; ++i)
        {
            if (*(exp_true_first + i) != *(host_vals.get() + i))
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "TRUE> got: " << *(host_vals.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_true_first + i) << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
        }

        for (int i = 0; i < ::std::min(exp.second - exp_false_first, res.second - first3) && is_correct; ++i)
        {
            if (*(exp_false_first + i) != *(host_res.get() + i))
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "FALSE> got: " << *(host_res.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_false_first + i) << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
        }

        EXPECT_TRUE(is_correct, "wrong effect from partition_copy");
    }
};

DEFINE_TEST(test_is_heap_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap_until)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        // first element is always a heap
        EXPECT_TRUE(actual == first + 1, "wrong result of is_heap_until_1");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == (first + n / 2), "wrong result of is_heap_until_2");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == last, "wrong result of is_heap_until_3");
    }
};

DEFINE_TEST(test_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_merge)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;
        typedef typename ::std::iterator_traits<Iterator3>::value_type T3;

        auto value = T1(0);
        auto x = n > 1 ? n / 2 : n;
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        ::std::iota(host_vals.get(), host_vals.get() + n, T2(value));
        update_data(host_keys, host_vals);

        ::std::vector<T3> exp(2 * n);
        auto exp1 = ::std::merge(host_keys.get(), host_keys.get() + n, host_vals.get(), host_vals.get() + x, exp.begin());
        auto res1 = ::std::merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2 + x, first3);
        TestDataTransfer<UDTKind::eRes, Size> host_res(*this, res1 - first3);
        wait_and_throw(exec);

        // Special case, because we have more results then source data
        host_res.retrieve_data();
        auto host_first3 = host_res.get();
#if _ONEDPL_DEBUG_SYCL
        for (size_t i = 0; i < res1 - first3; ++i)
        {
            if (host_first3[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_first3[i] << ::std::endl;
            }
        }
#endif // _ONEDPL_DEBUG_SYCL

        EXPECT_TRUE(res1 - first3 == exp1 - exp.begin(), "wrong result from merge_1");
        EXPECT_TRUE(::std::is_sorted(host_first3, host_first3 + (res1 - first3)), "wrong effect from merge_1");
    }
};

DEFINE_TEST(test_stable_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_stable_sort)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(333);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
        wait_and_throw(exec);

        {
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();

#if _ONEDPL_DEBUG_SYCL
            for (int i = 0; i < n; ++i)
            {
                if (host_first1[i] != value + i)
                {
                    ::std::cout << "Error_1. i = " << i << ", expected = " << value + i << ", got = " << host_first1[i]
                              << ::std::endl;
                }
            }
#endif // _ONEDPL_DEBUG_SYCL

            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from stable_sort_1");
        }

        ::std::stable_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
#if _ONEDPL_DEBUG_SYCL
        for (int i = 0; i < n; ++i)
        {
            if (host_first1[i] != value + n - 1 - i)
            {
                ::std::cout << "Error_2. i = " << i << ", expected = " << value + n - 1 - i
                            << ", got = " << host_first1[i] << ::std::endl;
            }
        }
#endif // _ONEDPL_DEBUG_SYCL

        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()),
                    "wrong effect from stable_sort_3");
    }
};

DEFINE_TEST(test_includes)
{
    DEFINE_TEST_CONSTRUCTOR(test_includes)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));

        //first test case
        last1 = first1 + na;
        last2 = first2 + nb;

        ::std::copy(a, a + na, host_keys.get());
        ::std::copy(b, b + nb, host_vals.get());
        host_keys.update_data(na);
        host_vals.update_data(nb);

        auto result = ::std::includes(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2);
        wait_and_throw(exec);

        EXPECT_TRUE(result, "wrong effect from includes a, b");

        host_vals.retrieve_data();
        ::std::copy(c, c + nc, host_vals.get());
        host_vals.update_data(nc);

        result = ::std::includes(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        wait_and_throw(exec);

        EXPECT_TRUE(!result, "wrong effect from includes a, c");
    }
};

DEFINE_TEST(test_set_intersection)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_intersection)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        //first test case
        last1 = first1 + na;
        last2 = first2 + nb;
        ::std::copy(a, a + na, host_keys.get());
        ::std::copy(b, b + nb, host_vals.get());
        host_keys.update_data(na);
        host_vals.update_data(nb);

        last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2,
                                      first3);
        wait_and_throw(exec);

        host_res.retrieve_data();
        auto nres = last3 - first3;

        EXPECT_TRUE(nres == 6, "wrong size of intersection of a, b");

        auto result = ::std::includes(host_keys.get(), host_keys.get() + na, host_res.get(), host_res.get() + nres) &&
                      ::std::includes(host_vals.get(), host_vals.get() + nb, host_res.get(), host_res.get() + nres);
        wait_and_throw(exec);

        EXPECT_TRUE(result, "wrong effect from set_intersection a, b");

        { //second test case

            last2 = first2 + nd;
            ::std::copy(a, a + na, host_keys.get());
            ::std::copy(d, d + nd, host_vals.get());
            host_keys.update_data(na);
            host_vals.update_data(nb);

            last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                          last2, first3);
            wait_and_throw(exec);

            auto nres = last3 - first3;
            EXPECT_TRUE(nres == 0, "wrong size of intersection of a, d");
        }
    }
};

DEFINE_TEST(test_set_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_difference)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        last1 = first1 + na;
        last2 = first2 + nb;

        ::std::copy(a, a + na, host_keys.get());
        ::std::copy(b, b + nb, host_vals.get());
        host_keys.update_data(na);
        host_vals.update_data(nb);

        last3 = ::std::set_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[na];
        host_res.retrieve_data();
        auto nres_expect = ::std::set_difference(host_keys.get(), host_keys.get() + na, host_vals.get(), host_vals.get() + nb, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_difference a, b");
    }
};

DEFINE_TEST(test_set_union)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_union)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        last1 = first1 + na;
        last2 = first2 + nb;

        ::std::copy(a, a + na, host_keys.get());
        ::std::copy(b, b + nb, host_vals.get());
        host_keys.update_data(na);
        host_vals.update_data(nb);

        last3 = ::std::set_union(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[na + nb];
        host_res.retrieve_data();
        auto nres_expect =
            ::std::set_union(host_keys.get(), host_keys.get() + na, host_vals.get(), host_vals.get() + nb, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_union a, b");
    }
};

DEFINE_TEST(test_set_symmetric_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_symmetric_difference)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes, Size>  host_res (*this, get_size(n));

        last1 = first1 + na;
        last2 = first2 + nb;

        ::std::copy(a, a + na, host_keys.get());
        ::std::copy(b, b + nb, host_vals.get());
        host_keys.update_data(na);
        host_vals.update_data(nb);

        last3 = ::std::set_symmetric_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                                                first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[na + nb];
        retrieve_data(host_keys, host_vals, host_res);
        auto nres_expect = ::std::set_symmetric_difference(host_keys.get(), host_keys.get() + na, host_vals.get(),
                                                           host_vals.get() + nb, res_expect) -
                           res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_symmetric_difference a, b");
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    //test3buffers
    PRINT_DEBUG("test_partition_copy");
    test3buffers<alloc_type, test_partition_copy<ValueType>>();
    PRINT_DEBUG("test_set_symmetric_difference");
    test3buffers<alloc_type, test_set_symmetric_difference<ValueType>>();
    PRINT_DEBUG("test_set_union");
    test3buffers<alloc_type, test_set_union<ValueType>>();
    PRINT_DEBUG("test_set_difference");
    test3buffers<alloc_type, test_set_difference<ValueType>>();
    PRINT_DEBUG("test_set_intersection");
    test3buffers<alloc_type, test_set_intersection<ValueType>>();
    PRINT_DEBUG("test_merge");
    test3buffers<alloc_type, test_merge<ValueType>>(2);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
    try
    {
#if TEST_DPCPP_BACKEND_PRESENT
        //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
        //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

        // Run tests for USM shared memory
        test_usm_and_buffer<sycl::usm::alloc::shared>();
        // Run tests for USM device memory
        test_usm_and_buffer<sycl::usm::alloc::device>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
