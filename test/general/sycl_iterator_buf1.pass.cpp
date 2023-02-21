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

DEFINE_TEST(test_destroy)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1 + (n / 3),
                       first1 + (n / 2));
        if (!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from destroy");
    }
};

DEFINE_TEST(test_destroy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy_n(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1, n);
        if(!::std::is_trivially_destructible<T1>::value)
            value = T1{-2};
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value),
                    "wrong effect from destroy_n");
    }
};

DEFINE_TEST(test_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), value);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value), "wrong effect from fill");
    }
};

DEFINE_TEST(test_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1), "wrong effect from fill_n");
    }
};

DEFINE_TEST(test_generate)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                      Generator_count<T1>(value));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from generate");
    }
};

DEFINE_TEST(test_generate_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Generator_count<T1>(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from generate_n");
    }
};

DEFINE_TEST(test_for_each)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value - 1);
        host_keys.update_data();

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), Inc());
        wait_and_throw(exec);

        // We call due to SYCL 1.2.1: 4.7.2.3.
        // If the host memory is modified by the host,
        // or mapped to another buffer or image during the lifetime of this buffer,
        // then the results are undefined
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value), "wrong effect from for_each");
    }
};

DEFINE_TEST(test_for_each_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each_n)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::for_each_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Inc());
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from for_each_n");
    }
};

DEFINE_TEST(test_replace)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, value, T1(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace");
    }
};

DEFINE_TEST(test_replace_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_if)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                          oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace_if");
    }
};

DEFINE_TEST(test_reduce)
{
    DEFINE_TEST_CONSTRUCTOR(test_reduce)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value);
        host_keys.update_data();

        // without initial value
        auto result1 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        EXPECT_TRUE(result1 == value * (n / 2 - n / 3), "wrong effect from reduce (1)");

        // with initial value
        auto init = T1(42);
        auto result2 = ::std::reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), init);
        wait_and_throw(exec);

        EXPECT_TRUE(result2 == init + value * (n / 2 - n / 3), "wrong effect from reduce (2)");
    }
};

DEFINE_TEST(test_transform_reduce_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_unary)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(1);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto result = ::std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, T1(42),
                                            Plus(), ::std::negate<T1>());
        wait_and_throw(exec);

        EXPECT_TRUE(result == 42 - n, "wrong effect from transform_reduce (unary + binary)");
    }
};

DEFINE_TEST(test_min_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_min_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{ static_cast<IteratorValueType>(::std::distance(first, last)) };

        ::std::for_each(host_keys.get(), host_keys.get() + n,
            [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t min_dis = n;
        if (min_dis)
        {
            *(host_keys.get() + min_dis / 2) = IteratorValueType{/*min_val*/ 0 };
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd min*/ 0 };
        }
        host_keys.update_data();

        auto result_min = ::std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();

        auto expected_min = ::std::min_element(host_keys.get(), host_keys.get() + n);

        EXPECT_TRUE(result_min - first == expected_min - host_keys.get(),
                    "wrong effect from min_element");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << *(host_keys.get() + (result_min - first)) << "["
                    << result_min - first << "], "
                    << "expected: " << *(host_keys.get() + (expected_min - host_keys.get())) << "["
                    << expected_min - host_keys.get() << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
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

DEFINE_TEST(test_max_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_max_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t max_dis = n;
        if (max_dis)
        {
            *(host_keys.get() + max_dis / 2) = IteratorValueType{/*max_val*/ 777};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd max*/ 777};
        }
        host_keys.update_data();

        auto expected_max_offset = ::std::max_element(host_keys.get(), host_keys.get() + n) - host_keys.get();

        auto result_max_offset = ::std::max_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last) - first;
        wait_and_throw(exec);

        host_keys.retrieve_data();

        EXPECT_TRUE(result_max_offset == expected_max_offset, "wrong effect from max_element");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: "      << *(host_keys.get() + result_max_offset)   << "[" << result_max_offset   << "], "
                    << "expected: " << *(host_keys.get() + expected_max_offset) << "[" << expected_max_offset << "]"
                    << ::std::endl;
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

DEFINE_TEST(test_minmax_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_minmax_element)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto fill_value = IteratorValueType{ 0 };

        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](IteratorValueType& it) { it = fill_value++ % 10 + 1; });
        ::std::size_t dis = n;
        if (dis > 1)
        {
            auto min_it = host_keys.get() + /*min_idx*/ dis / 2 - 1;
            *(min_it) = IteratorValueType{/*min_val*/ 0 };

            auto max_it = host_keys.get() + /*max_idx*/ dis / 2;
            *(max_it) = IteratorValueType{/*max_val*/ 777 };
        }
        host_keys.update_data();

        auto expected = ::std::minmax_element(host_keys.get(), host_keys.get() + n);
        auto expected_min = expected.first - host_keys.get();
        auto expected_max = expected.second - host_keys.get();
        ::std::pair<Size, Size> expected_offset = { expected_min, expected_max };

        auto result = ::std::minmax_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        auto result_min = result.first - first;
        auto result_max = result.second - first;

        wait_and_throw(exec);

        EXPECT_TRUE(result_min == expected_min && result_max == expected_max, "wrong effect from minmax_element");
        if (!(result_min == expected_min && result_max == expected_max))
        {
            host_keys.retrieve_data();

            auto got_min = host_keys.get() + (result.first - first);
            auto got_max = host_keys.get() + (result.second - first);
            ::std::cout << "MIN got: " << got_min << "[" << result_min << "], "
                        << "expected: " << *(host_keys.get() + expected_offset.first) << "[" << expected_min << "]" << ::std::endl;
            ::std::cout << "MAX got: " << got_max << "[" << result_max << "], "
                        << "expected: " << *(host_keys.get() + expected_offset.second) << "[" << expected_max << "]" << ::std::endl;
        }
    }
};

DEFINE_TEST(test_is_sorted)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{ 0 };
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        bool expected_bool = true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_sorted (Test #1 sorted sequence)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #2 unsorted sequence - the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #3 unsorted sequence - the middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element (no predicate)
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #4 unsorted sequence - the middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the first element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted Test #5 unsorted sequence - the first element");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_count)
{
    DEFINE_TEST_CONSTRUCTOR(test_count)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{0});
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count (Test #1 arbitrary to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check when none should be counted
        expected = 0;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{12});
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count (Test #2 none to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check when all should be counted
        ::std::fill(host_keys.get(), host_keys.get() + n, ValueType{7});
        host_keys.update_data();

        expected = n;
        result = ::std::count(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, ValueType{7});
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count (Test #3 all to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_count_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_count_if)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last,
                                            [](ValueType const& value) { return value % 10 == 0; });
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #1 arbitrary to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check when none should be counted
        expected = 0;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last,
                                 [](ValueType const& value) { return value > 10; });
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #2 none to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check when all should be counted
        expected = n;
        result = ::std::count_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                 [](ValueType const& value) { return value < 10; });
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from count_if (Test #3 all to count)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got " << result << ", expected " << expected << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_is_partitioned)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_partitioned)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        if (n < 2)
            return;

        auto less_than = [](const ValueType& value) -> bool { return value < 10; };
        auto is_odd = [](const ValueType& value) -> bool { return value % 2; };

        bool expected_bool_less_then = false;
        bool expected_bool_is_odd = false;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = ++fill_value; });
        expected_bool_less_then = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, less_than);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, less_than);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_less_then, "wrong effect from is_partitioned (Test #1 less than)");

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_is_odd, "wrong effect from is_partitioned (Test #2 is odd)");

        // The code as below was added to prevent accessor destruction working with host memory
        ::std::partition(host_keys.get(), host_keys.get() + n, is_odd);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_is_odd,
                    "wrong effect from is_partitioned (Test #3 is odd after partition)");
    }
};

DEFINE_TEST(test_any_all_none_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_any_all_none_of)

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
            auto res0 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1,
                                      [n](T1 x) { return x == n - 1; });
            wait_and_throw(exec);

            EXPECT_TRUE(!res0, "wrong effect from any_of_0");
            res0 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1,
                                  [](T1 x) { return x == -1; });
            wait_and_throw(exec);

            EXPECT_TRUE(res0, "wrong effect from none_of_0");
            res0 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, first1,
                                 [](T1 x) { return x % 2 == 0; });
            wait_and_throw(exec);

            EXPECT_TRUE(res0, "wrong effect from all_of_0");
        }
        // any_of
        auto res1 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1,
                                  [n](T1 x) { return x == n - 1; });
        wait_and_throw(exec);

        EXPECT_TRUE(res1, "wrong effect from any_of_1");
        auto res2 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1,
                                  [](T1 x) { return x == -1; });
        wait_and_throw(exec);

        EXPECT_TRUE(!res2, "wrong effect from any_of_2");
        auto res3 = ::std::any_of(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1,
                                  [](T1 x) { return x % 2 == 0; });
        wait_and_throw(exec);

        EXPECT_TRUE(res3, "wrong effect from any_of_3");

        //none_of
        auto res4 = ::std::none_of(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1,
                                   [](T1 x) { return x == -1; });
        wait_and_throw(exec);

        EXPECT_TRUE(res4, "wrong effect from none_of");

        //all_of
        auto res5 = ::std::all_of(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1,
                                  [](T1 x) { return x % 2 == 0; });
        wait_and_throw(exec);

        EXPECT_TRUE(n == 1 || !res5, "wrong effect from all_of");
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

DEFINE_TEST(test_partition)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{ 0 });
        host_keys.update_data();

        // invoke partition
        auto unary_op = [](IteratorValueType value) { return (value % 3 == 0) && (value % 2 == 0); };
        auto res = ::std::partition(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, unary_op);
        wait_and_throw(exec);

        // check
        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op),
                    "wrong effect from partition");
        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{0});
        host_keys.update_data();

        // invoke stable_partition
        res = ::std::stable_partition(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, unary_op);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op) &&
                        ::std::is_sorted(host_keys.get(), host_keys.get() + (res - first)) &&
                        ::std::is_sorted(host_keys.get() + (res - first), host_keys.get() + n),
                    "wrong effect from stable_partition");
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

DEFINE_TEST(test_is_heap)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        // True only when n == 1
        wait_and_throw(exec);

        EXPECT_TRUE(actual == (n == 1), "wrong result of is_heap_11");

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, first);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_12");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == false, "wrong result of is_heap_21");

        auto end = first + n / 2;
        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, end);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_22");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_3");
    }
};

DEFINE_TEST(test_inplace_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_inplace_merge)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;
        auto value = T(0);

        ::std::iota(host_keys.get(), host_keys.get() + n, value);

        ::std::vector<T> exp(n);
        ::std::iota(exp.begin(), exp.end(), value);

        auto middle = ::std::stable_partition(host_keys.get(), host_keys.get() + n, [](const T& x) { return x % 2; });
        host_keys.update_data();

        ::std::inplace_merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + (middle - host_keys.get()), last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        for (size_t i = 0; i < n; ++i)
        {
            if (host_keys.get()[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_keys.get()[i] << ::std::endl;
            }
            EXPECT_TRUE(host_keys.get()[i] == exp[i], "wrong effect from inplace_merge");
        }
    }
};

DEFINE_TEST(test_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_sort)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(333);
        ::std::iota(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1);
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

            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from sort_1");
        }

        ::std::sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, ::std::greater<T1>());
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

        EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n, ::std::greater<T1>()), "wrong effect from sort_2");
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

DEFINE_TEST(test_reverse)
{
    DEFINE_TEST_CONSTRUCTOR(test_reverse)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::reverse(local_copy.begin(), local_copy.end());

        ::std::reverse(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from reverse");
    }
};

DEFINE_TEST(test_rotate)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from rotate");
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
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_for_each<ValueType>>();
    PRINT_DEBUG("test_for_each_n");
    test1buffer<alloc_type, test_for_each_n<ValueType>>();
    PRINT_DEBUG("test_replace");
    test1buffer<alloc_type, test_replace<ValueType>>();
    PRINT_DEBUG("test_replace_if");
    test1buffer<alloc_type, test_replace_if<ValueType>>();
    PRINT_DEBUG("test_fill");
    test1buffer<alloc_type, test_fill<ValueType>>();
    PRINT_DEBUG("test_fill_n");
    test1buffer<alloc_type, test_fill_n<ValueType>>();
    PRINT_DEBUG("test_generate");
    test1buffer<alloc_type, test_generate<ValueType>>();
    PRINT_DEBUG("test_generate_n");
    test1buffer<alloc_type, test_generate_n<ValueType>>();
    PRINT_DEBUG("test_reduce");
    test1buffer<alloc_type, test_reduce<ValueType>>();
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<alloc_type, test_transform_reduce_unary<ValueType>>();
    PRINT_DEBUG("test_any_all_none_of");
    test1buffer<alloc_type, test_any_all_none_of<ValueType>>();
    PRINT_DEBUG("test_is_sorted");
    test1buffer<alloc_type, test_is_sorted<ValueType>>();
    PRINT_DEBUG("test_count");
    test1buffer<alloc_type, test_count<ValueType>>();
    PRINT_DEBUG("test_count_if");
    test1buffer<alloc_type, test_count_if<ValueType>>();
    PRINT_DEBUG("test_is_partitioned");
    test1buffer<alloc_type, test_is_partitioned<ValueType>>();
    PRINT_DEBUG("test_sort");
    test1buffer<alloc_type, test_sort<ValueType>>();
    PRINT_DEBUG("test_min_element");
    test1buffer<alloc_type, test_min_element<ValueType>>();
    PRINT_DEBUG("test_max_element");
    test1buffer<alloc_type, test_max_element<ValueType>>();
    PRINT_DEBUG("test_minmax_element");
    test1buffer<alloc_type, test_minmax_element<ValueType>>();
    PRINT_DEBUG("test_inplace_merge");
    test1buffer<alloc_type, test_inplace_merge<ValueType>>();
    PRINT_DEBUG("test_reverse");
    test1buffer<alloc_type, test_reverse<ValueType>>();
    PRINT_DEBUG("test_rotate");
    test1buffer<alloc_type, test_rotate<ValueType>>();
    PRINT_DEBUG("test_partition");
    test1buffer<alloc_type, test_partition<ValueType>>();
    PRINT_DEBUG("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();
    PRINT_DEBUG("test_find_if");
    test1buffer<alloc_type, test_find_if<ValueType>>();
    PRINT_DEBUG("test_adjacent_find");
    test1buffer<alloc_type, test_adjacent_find<ValueType>>();
    PRINT_DEBUG("test_is_sorted_until");
    test1buffer<alloc_type, test_is_sorted_until<ValueType>>();
    PRINT_DEBUG("test_search_n");
    test1buffer<alloc_type, test_search_n<ValueType>>();
    PRINT_DEBUG("test_remove");
    test1buffer<alloc_type, test_remove<ValueType>>();
    PRINT_DEBUG("test_remove_if");
    test1buffer<alloc_type, test_remove_if<ValueType>>();
    PRINT_DEBUG("test_stable_sort");
    test1buffer<alloc_type, test_stable_sort<ValueType>>();
    PRINT_DEBUG("test_unique");
    test1buffer<alloc_type, test_unique<ValueType>>();
    PRINT_DEBUG("test_is_heap_until");
    test1buffer<alloc_type, test_is_heap_until<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill");
    test1buffer<alloc_type, test_uninitialized_fill<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill_n");
    test1buffer<alloc_type, test_uninitialized_fill_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_default_construct");
    test1buffer<alloc_type, test_uninitialized_default_construct<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_default_construct_n");
    test1buffer<alloc_type, test_uninitialized_default_construct_n<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_value_construct");
    test1buffer<alloc_type, test_uninitialized_value_construct<ValueType>>();
    PRINT_DEBUG("test_uninitialized_value_construct_n");
    test1buffer<alloc_type, test_uninitialized_value_construct_n<ValueType>>();
    print_debug("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();
    PRINT_DEBUG("test_destroy");
    test1buffer<alloc_type, test_destroy<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_destroy_n");
    test1buffer<alloc_type, test_destroy_n<SyclTypeWrapper<ValueType>>>();
    test1buffer<alloc_type, test_destroy_n<ValueType>>();
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
