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

DEFINE_TEST(test_find_first_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_first_of, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        // Reset values after previous execution
        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        if (n < 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto res =
                ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1, "Wrong effect from find_first_of_1");
        }
        else if (n >= 2 && n < 10)
        {
            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1,
                                            first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_2");

            // No matches
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_3");
        }
        else if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto pos1 = n / 5;
            auto pos2 = 3 * n / 5;
            auto num = 3;

            ::std::iota(host_keys.get() + pos2, host_keys.get() + pos2 + num, T1(7));
            host_keys.update_data();

            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1 + pos2, "Wrong effect from find_first_of_4");

            // Add second match
            ::std::iota(host_keys.get() + pos1, host_keys.get() + pos1 + num, T1(6));
            host_keys.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1 + pos1, "Wrong effect from find_first_of_5");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_find_first_of");
    test2buffers<alloc_type, test_find_first_of<ValueType>>();
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
