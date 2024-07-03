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

DEFINE_TEST(test_is_sorted_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted_until, 2.0f, 0.65f)

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

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_is_sorted_until");
    test1buffer<alloc_type, test_is_sorted_until<ValueType>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
    try
    {
#if TEST_DPCPP_BACKEND_PRESENT
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
