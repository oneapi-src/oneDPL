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

DEFINE_TEST(test_is_heap)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        // True only when n == 1
        wait_and_throw(exec);

        EXPECT_TRUE(actual == (n == 1), "wrong result of is_heap_11");

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, first);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_12");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == false, "wrong result of is_heap_21");

        auto end = first + n / 2;
        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, end);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_22");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_3");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();
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
