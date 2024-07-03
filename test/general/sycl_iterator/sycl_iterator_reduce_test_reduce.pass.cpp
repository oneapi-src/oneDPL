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

DEFINE_TEST(test_reduce)
{
    DEFINE_TEST_CONSTRUCTOR(test_reduce, 2.0f, 0.80f)

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

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_reduce");
    test1buffer<alloc_type, test_reduce<ValueType>>();
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
