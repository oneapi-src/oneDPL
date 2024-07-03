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

DEFINE_TEST(test_max_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_max_element, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t max_dis = n;
        if (max_dis)
        {
            *(host_keys.get() + max_dis / 2) = IteratorValueType{/*max_val*/ 777};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd max*/ 777};
        }
        host_keys.update_data();

        auto expected_max_offset = ::std::max_element(host_keys.get(), host_keys.get() + n) - host_keys.get();

        auto result_max_offset = ::std::max_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last) - first;
        wait_and_throw(exec);

        host_keys.retrieve_data();

        EXPECT_TRUE(result_max_offset == expected_max_offset, "wrong effect from max_element");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: "      << *(host_keys.get() + result_max_offset)   << "[" << result_max_offset   << "], "
                    << "expected: " << *(host_keys.get() + expected_max_offset) << "[" << expected_max_offset << "]"
                    << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_max_element");
    test1buffer<alloc_type, test_max_element<ValueType>>();
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
