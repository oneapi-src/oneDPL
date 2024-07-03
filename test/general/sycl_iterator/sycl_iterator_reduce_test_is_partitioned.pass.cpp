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

DEFINE_TEST(test_is_partitioned)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_partitioned, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        if (n < 2)
            return;

        auto less_than = [](const ValueType& value) -> bool { return value < 10; };
        auto is_odd = [](const ValueType& value) -> bool { return value % 2; };

        bool expected_bool_less_then = false;
        bool expected_bool_is_odd = false;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = ++fill_value; });
        expected_bool_less_then = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, less_than);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, less_than);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_less_then, "wrong effect from is_partitioned (Test #1 less than)");

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_is_odd, "wrong effect from is_partitioned (Test #2 is odd)");

        // The code as below was added to prevent accessor destruction working with host memory
        ::std::partition(host_keys.get(), host_keys.get() + n, is_odd);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        result_bool = ::std::is_partitioned(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool_is_odd,
                    "wrong effect from is_partitioned (Test #3 is odd after partition)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test1buffer<alloc_type, test_is_partitioned<ValueType>>();
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
