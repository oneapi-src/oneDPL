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

constexpr int a[] = {0, 0, 1, 1, 2, 6, 6, 9, 9};
constexpr int b[] = {0, 1, 1, 6, 6, 9};
constexpr int c[] = {0, 1, 6, 6, 6, 9, 9};
constexpr int d[] = {7, 7, 7, 8};
constexpr auto a_size = sizeof(a) / sizeof(a[0]);
constexpr auto b_size = sizeof(b) / sizeof(b[0]);
constexpr auto c_size = sizeof(c) / sizeof(c[0]);
constexpr auto d_size = sizeof(d) / sizeof(d[0]);

template <typename Size>
Size
get_size(Size n)
{
    return n + a_size + b_size + c_size + d_size;
}

DEFINE_TEST(test_set_intersection)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_intersection, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        //first test case
        last1 = first1 + a_size;
        last2 = first2 + b_size;
        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2,
                                      first3);
        wait_and_throw(exec);

        host_res.retrieve_data();
        auto nres = last3 - first3;

        EXPECT_TRUE(nres == 6, "wrong size of intersection of a, b");

        auto result = ::std::includes(host_keys.get(), host_keys.get() + a_size, host_res.get(), host_res.get() + nres) &&
                      ::std::includes(host_vals.get(), host_vals.get() + b_size, host_res.get(), host_res.get() + nres);
        wait_and_throw(exec);

        EXPECT_TRUE(result, "wrong effect from set_intersection a, b");

        { //second test case

            last2 = first2 + d_size;
            ::std::copy(a, a + a_size, host_keys.get());
            ::std::copy(d, d + d_size, host_vals.get());
            host_keys.update_data(a_size);
            host_vals.update_data(b_size);

            last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                          last2, first3);
            wait_and_throw(exec);

            auto nres = last3 - first3;
            EXPECT_TRUE(nres == 0, "wrong size of intersection of a, d");
        }
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test3buffers<alloc_type, test_set_intersection<ValueType>>();
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
