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

DEFINE_TEST(test_partial_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_partial_sort, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* first1 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        if (n <= 1)
            return;

        auto value = T1(333);
        auto init = value;
        ::std::generate(host_keys.get(), host_keys.get() + n, [&init]() { return init--; });
        host_keys.update_data();

        auto end_idx = ((n < 3) ? 1 : n / 3);
        // Sort a subrange
        {
            auto end1 = first1 + end_idx;
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, end1, last1);
            wait_and_throw(exec);

            // Make sure that elements up to end are sorted and remaining elements are bigger
            // than the last sorted one.
            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + end_idx), "wrong effect from partial_sort_1");

            auto res = ::std::all_of(host_first1 + end_idx, host_first1 + n,
                                   [&](T1 val) { return val >= *(host_first1 + end_idx - 1); });
            EXPECT_TRUE(res, "wrong effect from partial_sort_1");
        }

        // Sort a whole sequence
        if (end_idx > last1 - first1)
        {
            ::std::partial_sort(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, last1);
            wait_and_throw(exec);

            host_keys.retrieve_data();
            auto host_first1 = host_keys.get();
            EXPECT_TRUE(::std::is_sorted(host_first1, host_first1 + n), "wrong effect from partial_sort_2");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_partial_sort");
    test2buffers<alloc_type, test_partial_sort<ValueType>>();
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
