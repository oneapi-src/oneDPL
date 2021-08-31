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
#include <CL/sycl.hpp>

using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

struct test_inclusive_scan_by_segment
{
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, ... };
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        const ValT value = 1;

        int segment_length = 1;
        int i = 0;
        while (i != n)
        {
          for (int j = 0; j != 4*segment_length && i != n; ++j)
          {
              *host_keys = j/segment_length + 1;
              *host_vals = 1;
              *host_val_res = 0;
              ++host_keys;
              ++host_vals;
              ++host_val_res;
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
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        int segment_length = 1;
        ValT expected_segment_sum = segment_length * (segment_length + 1) / 2;
        KeyT current_key = *host_keys;
        ValT current_sum = 0;
        for (int i = 0; i != n; ++i)
        {
            if (current_key == *host_keys)
            {
              current_sum += *val_res;
              ++host_keys;
              ++val_res;
            } else {
                EXPECT_TRUE(current_sum == expected_segment_sum, "wrong effect from exclusive_scan_by_segment");
                current_sum = *val_res;
                current_key = *host_keys;
                if (current_key == 1) {
                    ++segment_length;
                    expected_segment_sum = segment_length * (segment_length + 1) / 2;
                }
                ++host_keys;
                ++val_res;
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
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        auto host_keys = get_host_pointer(keys_first);
        auto host_vals = get_host_pointer(vals_first);
        auto host_val_res = get_host_pointer(val_res_first);

        initialize_data(host_keys, host_vals, host_val_res, n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::inclusive_scan_by_segment(new_policy, keys_first, keys_last, vals_first, val_res_first);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys = get_host_pointer(keys_first);
        host_val_res = get_host_pointer(val_res_first);
        check_values(host_keys, host_val_res, n);

        // call algorithm with equality comparator
        host_vals = get_host_pointer(vals_first);

        initialize_data(host_keys, host_vals, host_val_res, n);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(new_policy2, keys_first, keys_last, vals_first, val_res_first,
                                                           ::std::equal_to<KeyT>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys = get_host_pointer(keys_first);
        host_val_res = get_host_pointer(val_res_first);
        check_values(host_keys, host_val_res, n);

        // call algorithm with addition operator
        host_vals = get_host_pointer(vals_first);

        initialize_data(host_keys, host_vals, host_val_res, n);

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(new_policy3, keys_first, keys_last, vals_first, val_res_first,
                                                           ::std::equal_to<KeyT>(), ::std::plus<ValT>());
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys = get_host_pointer(keys_first);
        host_val_res = get_host_pointer(val_res_first);
        check_values(host_keys, host_val_res, n);
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
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res1 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first);
        check_values(keys_first, vals_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           ::std::equal_to<KeyT>());
        check_values(keys_first, vals_first, n);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           ::std::equal_to<KeyT>(), ::std::plus<ValT>());
        check_values(keys_first, vals_first, n);

    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<is_same_iterator_category<Iterator3, ::std::bidirectional_iterator_tag>::value ||
                                  is_same_iterator_category<Iterator3, ::std::forward_iterator_tag>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
    }
};

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    test3buffers<uint64_t, test_inclusive_scan_by_segment>();
#endif
    return TestUtils::done();
}
