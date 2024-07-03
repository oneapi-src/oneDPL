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

DEFINE_TEST(test_stable_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_stable_sort, 2.0f, 0.65f)

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

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_stable_sort");
    test1buffer<alloc_type, test_stable_sort<ValueType>>();
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
