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

#include "sycl_iterator_test.h"

#if TEST_DPCPP_BACKEND_PRESENT

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

DEFINE_TEST(test_partial_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_partial_sort)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* first1 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        if (n <= 1)
            return;

        auto value = T1(333);
        auto init = value;
        ::std::generate(host_keys.get(), host_keys.get() + n, [&init]() { return init--; });
        host_keys.update_data();

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end1 = first1 + end_idx;
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, end1, last1);
            wait_and_throw(exec);

            // Make sure that elements up to end are sorted and remaining elements are bigger
            // than the last sorted one.
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + end_idx), "wrong effect from partial_sort_1");

            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n,
                                   [&](T1 val) { return val >= *(host_first1 + end_idx - 1); });
            EXPECT_TRUE(res, "wrong effect from partial_sort_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, last1);
            wait_and_throw(exec);

            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from partial_sort_2");
        }
    }
};

DEFINE_TEST(test_partial_sort_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_partial_sort_copy)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);

        if (n <= 1)
            return;

        auto init = value;
        ::std::generate(host_keys.get(), host_keys.get() + n, [&init]() { return init--; });
        host_keys.update_data();

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end2 = first2 + end_idx;

            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, end2);
            wait_and_throw(exec);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            EXPECT_TRUE(last_sorted == end2, "wrong effect from partial_sort_copy_1");
            // Make sure that elements up to end2 are sorted
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + end_idx), "wrong effect from partial_sort_copy_1");

            // Now ensure that the original sequence wasn't changed by partial_sort_copy
            auto init = value;
            auto res = ::std::all_of(host_first1, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            auto last_sorted =
                ::std::partial_sort_copy(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            EXPECT_TRUE(last_sorted == last2, "wrong effect from partial_sort_copy_2");
            EXPECT_TRUE(::std::is_sorted(host_first2, host_first2 + n), "wrong effect from partial_sort_copy_2");

            // Now ensure that partial_sort_copy hasn't change the unsorted part of original sequence
            auto init = value - end_idx;
            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n, [&init](T1 val) { return val == init--; });
            EXPECT_TRUE(res, "original sequence was changed by partial_sort_copy_2");
        }
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

DEFINE_TEST(test_nth_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_nth_element)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T1 = typename ::std::iterator_traits<Iterator1>::value_type;
        using T2 = typename ::std::iterator_traits<Iterator2>::value_type;

        // init
        auto value1 = T1(0);
        auto value2 = T2(0);
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&value1](T1& val) { val = (value1++ % 10) + 1; });
        ::std::for_each(host_vals.get(), host_vals.get() + n, [&value2](T2& val) { val = (value2++ % 10) + 1; });
        update_data(host_keys, host_vals);

        auto middle1 = first1 + n / 2;

        // invoke
        auto comp = ::std::less<T1>{};
        ::std::nth_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, middle1, last1, comp);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        ::std::nth_element(host_first2, host_first2 + n / 2, host_first2 + n, comp);

        // check
        auto median = *(host_first1 + n / 2);
        bool is_correct = median == *(host_first2 + n / 2);
        if (!is_correct)
        {
            ::std::cout << "wrong nth element value got: " << median << ", expected: " << *(host_first2 + n / 2)
                      << ::std::endl;
        }
        is_correct =
            ::std::find_first_of(host_first1, host_first1 + n / 2, host_first1 + n / 2, host_first1 + n,
                               [comp](T1& x, T2& y) { return comp(y, x); }) ==
                     host_first1 + n / 2;
        EXPECT_TRUE(is_correct, "wrong effect from nth_element");
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

#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_sort");
    test1buffer<alloc_type, test_sort<ValueType>>();
    PRINT_DEBUG("test_inplace_merge");
    test1buffer<alloc_type, test_inplace_merge<ValueType>>();
    PRINT_DEBUG("test_stable_sort");
    test1buffer<alloc_type, test_stable_sort<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_nth_element");
    test2buffers<alloc_type, test_nth_element<ValueType>>();
    PRINT_DEBUG("test_partial_sort");
    test2buffers<alloc_type, test_partial_sort<ValueType>>();
    PRINT_DEBUG("test_partial_sort_copy");
    test2buffers<alloc_type, test_partial_sort_copy<ValueType>>();

    //test3buffers
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
