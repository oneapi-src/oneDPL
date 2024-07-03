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

DEFINE_TEST(test_inplace_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_inplace_merge, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;
        auto value = T(0);

        ::std::iota(host_keys.get(), host_keys.get() + n, value);

        ::std::vector<T> exp(n);
        ::std::iota(exp.begin(), exp.end(), value);

        auto middle = ::std::stable_partition(host_keys.get(), host_keys.get() + n, [](const T& x) { return x % 2; });
        host_keys.update_data();

        ::std::inplace_merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + (middle - host_keys.get()), last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        for (size_t i = 0; i < n; ++i)
        {
            if (host_keys.get()[i] != exp[i])
            {
                ::std::cout << "Error: i = " << i << ", expected " << exp[i] << ", got " << host_keys.get()[i] << ::std::endl;
            }
            EXPECT_TRUE(host_keys.get()[i] == exp[i], "wrong effect from inplace_merge");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test1buffer<alloc_type, test_inplace_merge<ValueType>>();
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
