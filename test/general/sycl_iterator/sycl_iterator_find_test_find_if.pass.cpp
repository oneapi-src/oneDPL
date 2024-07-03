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

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if, 2.0f, 0.65f)

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

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test1buffer<alloc_type, test_find_if<ValueType>>();
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
