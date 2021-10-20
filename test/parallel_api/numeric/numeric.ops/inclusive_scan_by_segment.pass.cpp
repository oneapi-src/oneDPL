// -*- C++ -*-
//===-- inclusive_scan_by_segment.pass.cpp ------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"

using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

struct test_inclusive_scan_by_segment
{
    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, ... };

        int segment_length = 1;
        int i = 0;
        while (i != n)
        {
          for (int j = 0; j != 4*segment_length && i != n; ++j)
          {
              host_keys[i] = j/segment_length + 1;
              host_vals[i] = 1;
              host_val_res[i] = 0;
              ++i;
          }
          ++segment_length;
        }
    }

    template <typename Iterator1, typename Iterator2, typename Size>
    void
    check_values(Iterator1 host_keys, Iterator2 val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, ...};

        int segment_length = 1;
        auto expected_segment_sum = 1;
        auto current_key = host_keys[0];
        auto current_sum = 0;
        for (int i = 0; i != n; ++i)
        {
            if (current_key == host_keys[i])
            {
              current_sum += val_res[i];
            } else {
                EXPECT_TRUE(current_sum == expected_segment_sum, "wrong effect from exclusive_scan_by_segment");
                current_sum = val_res[i];
                current_key = host_keys[i];
                if (current_key == 1) {
                    ++segment_length;
                    expected_segment_sum = segment_length * (segment_length + 1) / 2;
                }
            }
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        {
            auto host_keys = get_host_access(keys_first);
            auto host_vals = get_host_access(vals_first);
            auto host_val_res = get_host_access(val_res_first);

            initialize_data(host_keys, host_vals, host_val_res, n);
        }

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::inclusive_scan_by_segment(new_policy, keys_first, keys_last, vals_first, val_res_first);
        exec.queue().wait_and_throw();

        {
            auto host_keys = get_host_access(keys_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_keys, host_val_res, n);

            // call algorithm with equality comparator
            auto host_vals = get_host_access(vals_first);

            initialize_data(host_keys, host_vals, host_val_res, n);
        }

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(new_policy2, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; });
        exec.queue().wait_and_throw();
        {
            auto host_keys = get_host_access(keys_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_keys, host_val_res, n);

            // call algorithm with addition operator
            auto host_vals = get_host_access(vals_first);

            initialize_data(host_keys, host_vals, host_val_res, n);
        }

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(new_policy3, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           [](ValT first, ValT second) { return first + second; });
        exec.queue().wait_and_throw();
        auto host_keys = get_host_access(keys_first);
        auto host_val_res = get_host_access(val_res_first);
        check_values(host_keys, host_val_res, n);
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res1 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first);
        check_values(keys_first, val_res_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; });
        check_values(keys_first, val_res_first, n);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           [](ValT first, ValT second) { return first + second; });
        check_values(keys_first, val_res_first, n);

    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<!is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
    }
};

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    test3buffers<sycl::usm::alloc::shared, uint64_t, test_inclusive_scan_by_segment>();
    test3buffers<sycl::usm::alloc::device, uint64_t, test_inclusive_scan_by_segment>();
#endif
    test_algo_three_sequences<uint64_t, test_inclusive_scan_by_segment>();
    return TestUtils::done();
}
