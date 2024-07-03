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

DEFINE_TEST(test_copy_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_if, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto res1 = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                                   [](T1 x) { return x > -1; });
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last2, "wrong result from copy_if_1");

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < res1 - first2; ++i)
        {
            auto exp = i + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_1");
        }

        auto res2 = ::std::copy_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                 [](T1 x) { return x % 2 == 1; });
        wait_and_throw(exec);

        EXPECT_TRUE(res2 == first2 + (last2 - first2) / 2, "wrong result from copy_if_2");

        host_vals.retrieve_data();
        host_first2 = host_vals.get();
        for (int i = 0; i < res2 - first2; ++i)
        {
            auto exp = 2 * i + 1 + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_2: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_2");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_copy_if");
    test2buffers<alloc_type, test_copy_if<ValueType>>();
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
