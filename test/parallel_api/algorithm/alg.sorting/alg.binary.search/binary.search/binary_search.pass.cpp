// -*- C++ -*-
//===-- binary_search.pass.cpp --------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>

using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

struct test_binary_search
{
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    initialize_data(Iterator1 data, Iterator2 value, Iterator3 result, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValT;
        int num_values = n * .01 > 1 ? n * .01 : 1; // # search values expected to be << n
        for (int i = 0; i < n; i += 2)
        {
            *data = i;
            ++data;
            if (i + 1 < n)
            {
                *data = i;
                ++data;
            }
            if (i < num_values * 2)
            {
                // value = {0, 2, 5, 6, 9, 10, 13...}
                // result will alternate true/false after initial true
                *value = i + (i != 0 && i % 4 == 0 ? 1 : 0);
                ++value;
            }
            *result = 0;
            ++result;
        }
    }

    template <typename Iterator1, typename Size>
    void
    check_values(Iterator1 result, Size n)
    {
        int num_values = n * .01 > 1 ? n * .01 : 1; // # search values expected to be << n
        for (int i = 0; i != num_values; ++i, ++result)
        {
            if (i == 0)
            {
                EXPECT_TRUE(*result == true, "wrong effect from binary_search");
            }
            else
            {
                EXPECT_TRUE(*result == i % 2, "wrong effect from binary_search");
            }
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            !is_same_iterator_category<Iterator3, ::std::bidirectional_iterator_tag>::value &&
            !is_same_iterator_category<Iterator3, ::std::forward_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator2 value_first, Iterator2 value_last,
               Iterator3 result_first, Iterator3 result_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValueT;

        // call algorithm with no optional arguments
        auto host_first = get_host_pointer(first);
        auto host_val_first = get_host_pointer(value_first);
        auto host_result = get_host_pointer(result_first);

        initialize_data(host_first, host_val_first, host_result, n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::binary_search(new_policy, first, last, value_first, value_last, result_first);
#    if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#    endif
        host_first = get_host_pointer(first);
        host_val_first = get_host_pointer(value_first);
        host_result = get_host_pointer(result_first);
        check_values(host_result, n);

        // call algorithm with comparator
        initialize_data(host_first, host_val_first, host_result, n);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::binary_search(new_policy2, first, last, value_first, value_last, result_first,
                                               ::std::less<ValueT>());
#    if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#    endif
        host_first = get_host_pointer(first);
        host_val_first = get_host_pointer(value_first);
        host_result = get_host_pointer(result_first);
        check_values(host_result, n);
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            !is_same_iterator_category<Iterator3, ::std::bidirectional_iterator_tag>::value &&
            !is_same_iterator_category<Iterator3, ::std::forward_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator2 value_first, Iterator2 value_last,
               Iterator3 result_first, Iterator3 result_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValueT;

        // call algorithm with no optional arguments
        initialize_data(first, value_first, result_first, n);

        auto res1 = oneapi::dpl::binary_search(exec, first, last, value_first, value_last, result_first);
        check_values(result_first, n);

        // call algorithm with comparator
        initialize_data(first, value_first, result_first, n);

        auto res2 =
            oneapi::dpl::binary_search(exec, first, last, value_first, value_last, result_first, ::std::less<ValueT>());
        check_values(result_first, n);
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<is_same_iterator_category<Iterator3, ::std::bidirectional_iterator_tag>::value ||
                                  is_same_iterator_category<Iterator3, ::std::forward_iterator_tag>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator2 value_first, Iterator2 value_last,
               Iterator3 result_first, Iterator3 result_last, Size n)
    {
    }
};

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test3buffers<uint64_t, test_binary_search>();
#endif
    return TestUtils::done();
}
