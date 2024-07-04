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

DEFINE_TEST(test_search)
{
    DEFINE_TEST_CONSTRUCTOR(test_search, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(5));
        ::std::iota(host_vals.get(), host_vals.get() + n, T1(0));
        update_data(host_keys, host_vals);

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from search_00");
            res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from search_01");
        }
        auto res1 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1 == last1, "wrong effect from search_1");
        if (n > 10)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::search(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 10, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res2 - first1 == 5, "wrong effect from search_2");
        }
        // subsequence consists of one element (last one)
        auto res3 = ::std::search(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, last1 - 1, last1);
        wait_and_throw(exec);

        EXPECT_TRUE(last1 - res3 == 1, "wrong effect from search_3");

        // first sequence contains 2 almost similar parts
        if (n > 5)
        {
            ::std::iota(host_keys.get() + n / 2, host_keys.get() + n, T1(5));
            host_keys.update_data();

            auto res4 = ::std::search(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2 + 5, first2 + 6);
            wait_and_throw(exec);

            EXPECT_TRUE(res4 == first1, "wrong effect from search_4");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test2buffers<alloc_type, test_search<ValueType>>();
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
