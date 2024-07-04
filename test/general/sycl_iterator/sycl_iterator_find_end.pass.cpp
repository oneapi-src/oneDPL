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

DEFINE_TEST(test_find_end)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_end, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;

        // Reset after previous run
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        }

        if (n <= 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            host_vals.update_data();

            // Empty subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_end_1");

            return;
        }

        if (n > 2 && n < 10)
        {
            // re-write the sequence after previous run
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            update_data(host_keys, host_vals);

            // No subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2 + n / 2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_end_2");

            // Whole sequence is matched
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(10));
            host_keys.update_data();

            res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1, "Wrong effect from find_end_3");

            return;
        }

        if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n / 5, T2(20));
            host_vals.update_data();

            // Match at the beginning
            {
                auto start = host_keys.get();
                auto end = host_keys.get() + n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1, "Wrong effect from find_end_4");
            }

            // 2 matches: at the beginning and in the middle, should return the latter
            {
                auto start = host_keys.get() + 2 * n / 5;
                auto end = host_keys.get() + 3 * n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();


                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1 + 2 * n / 5, "Wrong effect from find_end_5");
            }

            // 3 matches: at the beginning, in the middle and at the end, should return the latter
            {
                auto start = host_keys.get() + 4 * n / 5;
                auto end = host_keys.get() + n;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1 + 4 * n / 5, "Wrong effect from find_end_6");
            }
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test2buffers<alloc_type, test_find_end<ValueType>>();
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
