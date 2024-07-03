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

DEFINE_TEST(test_swap_ranges)
{
    DEFINE_TEST_CONSTRUCTOR(test_swap_ranges, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using value_type = typename ::std::iterator_traits<Iterator1>::value_type;
        using reference = typename ::std::iterator_traits<Iterator1>::reference;

        ::std::iota(host_keys.get(), host_keys.get() + n, value_type(0));
        ::std::iota(host_vals.get(), host_vals.get() + n, value_type(n));
        update_data(host_keys, host_vals);

        Iterator2 actual_return = ::std::swap_ranges(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);

        wait_and_throw(exec);

        bool check_return = (actual_return == last2);
        EXPECT_TRUE(check_return, "wrong result of swap_ranges");
        if (check_return)
        {
            ::std::size_t i = 0;

            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();
            bool check =
                ::std::all_of(host_first2, host_first2 + n, [&i](reference a) { return a == value_type(i++); }) &&
                ::std::all_of(host_first1, host_first1 + n, [&i](reference a) { return a == value_type(i++); });

            EXPECT_TRUE(check, "wrong effect of swap_ranges");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test2buffers<alloc_type, test_swap_ranges<ValueType>>();
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
