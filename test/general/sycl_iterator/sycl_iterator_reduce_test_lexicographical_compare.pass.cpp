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

// TODO: move unique cases to test_lexicographical_compare
DEFINE_TEST(test_lexicographical_compare)
{
    DEFINE_TEST_CONSTRUCTOR(test_lexicographical_compare, 2.0f, 0.80f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // INIT
        {
            ValueType fill_value1{0};
            ::std::for_each(host_keys.get(), host_keys.get() + n,
                            [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
            ValueType fill_value2{0};
            ::std::for_each(host_vals.get(), host_vals.get() + n,
                            [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
            update_data(host_keys, host_vals);
        }

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

        // CHECK 1.1: S1 == S2 && len(S1) == len(S2)
        bool is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1,
                                                          last1, first2, last2, comp);
        wait_and_throw(exec);

        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.1: S1 == S2 && len(S1) == len(S2)");

        // CHECK 1.2: S1 == S2 && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 1.2: S1 == S2 && len(S1) < len(S2)");

        // CHECK 1.3: S1 == S2 && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 1.3: S1 == S2 && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_vals.get() + n - 2) = 222;
            host_vals.update_data();
        }

        // CHECK 2.1: S1 < S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        bool is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.1: S1 < S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 2.2: S1 < S2 (PRE-LAST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 2.2: S1 < S2 (PRE-LAST) && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_keys.get() + n - 2) = 333;
            host_keys.update_data();
        }

        // CHECK 3.1: S1 > S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0,
                    "wrong effect from lex_compare Test 3.1: S1 > S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 3.2: S1 > S2 (PRE-LAST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 3.2: S1 > S2 (PRE-LAST) && len(S1) < len(S2)");
        {
            *host_vals.get() = 444;
            host_vals.update_data();
        }

        // CHECK 4.1: S1 < S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        if (is_less_res != 1)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 1" << ::std::endl;
        EXPECT_TRUE(is_less_res == 1, "wrong effect from lex_compare Test 4.1: S1 < S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 4.2: S1 < S2 (FIRST ELEMENT) && len(S1) > len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 8>>(exec), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 1 : 0;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 4.2: S1 < S2 (FIRST) && len(S1) > len(S2)");
        {
            *host_keys.get() = 555;
            host_keys.update_data();
        }

        // CHECK 5.1: S1 > S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 9>>(exec), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        if (is_less_res != 0)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected 0" << ::std::endl;
        EXPECT_TRUE(is_less_res == 0, "wrong effect from lex_compare Test 5.1: S1 > S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 5.2: S1 > S2 (FIRST ELEMENT) && len(S1) < len(S2)
        is_less_res = ::std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 10>>(exec), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 0 : 1;
        if (is_less_res != is_less_exp)
            ::std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << ::std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp,
                    "wrong effect from lex_compare Test 5.2: S1 > S2 (FIRST) && len(S1) < len(S2)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<alloc_type, test_lexicographical_compare<ValueType>>();
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
