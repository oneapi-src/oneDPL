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

DEFINE_TEST(test_rotate)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from rotate");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test1buffer<alloc_type, test_rotate<ValueType>>();
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
