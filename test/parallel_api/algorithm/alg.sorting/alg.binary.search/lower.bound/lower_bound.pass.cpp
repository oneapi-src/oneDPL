// -*- C++ -*-
//===-- lower_bound.pass.cpp --------------------------------------------===//
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
#include "../binary_search_utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#endif

#include <cmath>

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

struct test_lower_bound
{
    // TODO: replace data generation with random data and update check to compare result to
    // the result of the serial algorithm
    template <typename Accessor1, typename Accessor2, typename Size>
    void
    check_and_clean(Accessor1 result, Accessor2 value, Size n)
    {
        int num_values = n * .01 > 1 ? n * .01 : 1; // # search values expected to be << n
        for (int i = 0; i != num_values; ++i)
        {
            EXPECT_TRUE((std::ceil(value[i] / 2.)) * 2 == result[i], "wrong effect from lower_bound");
            // clean result for next test case
            result[i] = 0;
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator2 value_first, Iterator2 value_last,
               Iterator3 result_first, Iterator3 result_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValueT;
        // call algorithm with no optional arguments
        {
            auto host_first = get_host_access(first);
            auto host_val_first = get_host_access(value_first);
            auto host_result = get_host_access(result_first);

            initialize_data(host_first, host_val_first, host_result, n);
        }

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::lower_bound(new_policy, first, last, value_first, value_last, result_first);
        exec.queue().wait_and_throw();
        {
            auto host_val_first = get_host_access(value_first);
            auto host_result = get_host_access(result_first);
            check_and_clean(host_result, host_val_first, n);
        }

        // call algorithm with comparator
        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::lower_bound(new_policy2, first, last, value_first, value_last, result_first,
                                             [](ValueT first, ValueT second) { return first < second; });
        exec.queue().wait_and_throw();
        {
            auto host_val_first = get_host_access(value_first);
            auto host_result = get_host_access(result_first);
            check_and_clean(host_result, host_val_first, n);
        }
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator2 value_first, Iterator2 value_last,
               Iterator3 result_first, Iterator3 result_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type ValueT;
        // call algorithm with no optional arguments
        initialize_data(first, value_first, result_first, n);

        auto res1 = oneapi::dpl::lower_bound(exec, first, last, value_first, value_last, result_first);
        check_and_clean(result_first, value_first, n);

        // call algorithm with comparator
        auto res2 = oneapi::dpl::lower_bound(exec, first, last, value_first, value_last, result_first,
                                             [](ValueT first, ValueT second) { return first < second; });
        check_and_clean(result_first, value_first, n);
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<!is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
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
    test3buffers<sycl::usm::alloc::shared, uint64_t, test_lower_bound>();
    test3buffers<sycl::usm::alloc::device, uint64_t, test_lower_bound>();
#endif
    test_algo_three_sequences<uint64_t, test_lower_bound>();
    return TestUtils::done();
}
