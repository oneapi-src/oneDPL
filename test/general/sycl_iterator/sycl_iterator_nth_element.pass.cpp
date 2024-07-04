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

DEFINE_TEST(test_nth_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_nth_element, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T1 = typename ::std::iterator_traits<Iterator1>::value_type;
        using T2 = typename ::std::iterator_traits<Iterator2>::value_type;

        // init
        auto value1 = T1(0);
        auto value2 = T2(0);
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&value1](T1& val) { val = (value1++ % 10) + 1; });
        ::std::for_each(host_vals.get(), host_vals.get() + n, [&value2](T2& val) { val = (value2++ % 10) + 1; });
        update_data(host_keys, host_vals);

        auto middle1 = first1 + n / 2;

        // invoke
        auto comp = ::std::less<T1>{};
        ::std::nth_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, middle1, last1, comp);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        ::std::nth_element(host_first2, host_first2 + n / 2, host_first2 + n, comp);

        // check
        auto median = *(host_first1 + n / 2);
        bool is_correct = median == *(host_first2 + n / 2);
        if (!is_correct)
        {
            ::std::cout << "wrong nth element value got: " << median << ", expected: " << *(host_first2 + n / 2)
                      << ::std::endl;
        }
        is_correct =
            ::std::find_first_of(host_first1, host_first1 + n / 2, host_first1 + n / 2, host_first1 + n,
                               [comp](T1& x, T2& y) { return comp(y, x); }) ==
                     host_first1 + n / 2;
        EXPECT_TRUE(is_correct, "wrong effect from nth_element");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    test2buffers<alloc_type, test_nth_element<ValueType>>();
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
