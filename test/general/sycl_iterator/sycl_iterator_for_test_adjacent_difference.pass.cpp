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

DEFINE_TEST(test_adjacent_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_difference, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        Iterator1ValueType fill_value{1};
        Iterator2ValueType blank_value{0};

        auto __f = [](Iterator1ValueType& a, Iterator1ValueType& b) -> Iterator2ValueType { return a + b; };

        // init
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](Iterator1ValueType& val) { val = (fill_value++ % 10) + 1; });
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        update_data(host_keys, host_vals);

        // test with custom functor
        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, __f);
        wait_and_throw(exec);

        {
            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            bool is_correct = *host_first1 == *host_first2; // for the first element
            for (int i = 1; i < n; ++i)                     // for subsequent elements
                is_correct = is_correct && *(host_first2 + i) == __f(*(host_first1 + i), *(host_first1 + i - 1));

            EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #1");
        }

        // test with default functor
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        host_vals.update_data();

        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        bool is_correct = *host_first1 == *host_first2; // for the first element
        for (int i = 1; i < n; ++i)                     // for subsequent elements
            is_correct = is_correct && *(host_first2 + i) == (*(host_first1 + i) - *(host_first1 + i - 1));

        EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #2");
    }
};

#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_adjacent_difference");
    test2buffers<alloc_type, test_adjacent_difference<ValueType>>();
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
