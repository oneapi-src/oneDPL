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

DEFINE_TEST(test_equal)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = T(42);

        auto new_start = n / 3;
        auto new_end = n / 2;

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, T{0});
        ::std::fill(host_vals.get() + new_start, host_vals.get() + new_end, value);
        update_data(host_keys, host_vals);

        auto expected  = new_end - new_start > 0;
        auto result = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + new_start,
                                   first1 + new_end, first2 + new_start);
        wait_and_throw(exec);

        EXPECT_TRUE(expected == result, "wrong effect from equal with 3 iterators");
        result = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1 + new_start, first1 + new_end,
                              first2 + new_start, first2 + new_end);
        wait_and_throw(exec);

        EXPECT_TRUE(expected == result, "wrong effect from equal with 4 iterators");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test2buffers<alloc_type, test_equal<ValueType>>();
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
