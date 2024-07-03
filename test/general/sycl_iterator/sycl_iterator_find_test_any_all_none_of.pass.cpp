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

DEFINE_TEST(test_any_all_none_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_any_all_none_of, 2.0f, 0.65f)

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

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_any_all_none_of");
    test1buffer<alloc_type, test_any_all_none_of<ValueType>>();
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
