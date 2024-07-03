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

DEFINE_TEST(test_mismatch)
{
    DEFINE_TEST_CONSTRUCTOR(test_mismatch, 2.0f, 0.65f)

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
            auto res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_00");
            res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_01");
        }
        auto res1 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1.first == first1 && res1.second == first2, "wrong effect from mismatch_1");
        if (n > 5)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 5, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res2.first == last1 - 5 && res2.second == last2, "wrong effect from mismatch_2");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_mismatch");
    test2buffers<alloc_type, test_mismatch<ValueType>>();
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
