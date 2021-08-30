// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp --------------------------------------------===//
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
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>

using namespace oneapi::dpl::execution;
#endif
using namespace TestUtils;

struct test_reduce_by_segment
{
    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_key_res, Iterator4 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, ..., 0 };
        //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, ..., 0 };
        for (int i = 0; i < n - 1; ++i)
        {
            std::size_t fill_first = i + 1;
            std::size_t fill_repeat = i % 4 < 2 ? 1 : 3;
            std::size_t fill = i < 4 ? fill_first : fill_repeat;

            host_keys[i] = fill;
            host_vals[i] = fill;
            host_key_res[i] = 9;
            host_val_res[i] = 1;
        }
        host_keys[n - 1] = 0;
        host_vals[n - 1] = 0;
    }

    template <typename Iterator1, typename Iterator2, typename Size>
    void
    check_values(Iterator1 key_res, Iterator2 val_res, Size n)
    {
        // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, ..., 0};
        // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, ..., 0};

        for (auto i = 0; i != n; ++i)
        {
            if (i == n - 1)
            {
                EXPECT_TRUE(key_res[i] == 0, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(val_res[i] == 0, "wrong effect from reduce_by_segment");
            }
            else if (i < 4)
            {
                EXPECT_TRUE(key_res[i] == i + 1, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(val_res[i] == i + 1, "wrong effect from reduce_by_segment");
            }
            else if (i % 2 == 0)
            {
                EXPECT_TRUE(key_res[i] == 1, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(val_res[i] == 2 || (val_res[i] == 1 && i == n - 2), "wrong effect from reduce_by_segment");
            }
            else
            {
                EXPECT_TRUE(key_res[i] == 3, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(val_res[i] == 6 || val_res[i] == 3, "wrong effect from reduce_by_segment");
            }
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    typename ::std::enable_if<
        oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
            is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        {
            auto host_keys = get_host_access(keys_first);
            auto host_vals = get_host_access(vals_first);
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);

            initialize_data(host_keys, host_vals, host_key_res, host_val_res, n);
        }

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 =
            oneapi::dpl::reduce_by_segment(new_policy, keys_first, keys_last, vals_first, key_res_first, val_res_first);
        new_policy.queue().wait_and_throw();
        Size result_size = std::distance(key_res_first, res1.first);

        {
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_key_res, host_val_res, result_size);
        }

        // call algorithm with equality comparator
        {
            auto host_keys = get_host_access(keys_first);
            auto host_vals = get_host_access(vals_first);
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);

            initialize_data(host_keys, host_vals, host_key_res, host_val_res, n);
        }

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::reduce_by_segment(new_policy2, keys_first, keys_last, vals_first, key_res_first,
                                                   val_res_first, [](KeyT first, KeyT second) { return first == second; });
        new_policy2.queue().wait_and_throw();
        result_size = std::distance(key_res_first, res2.first);

        {
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_key_res, host_val_res, result_size);
        }

        // call algorithm with addition operator
        {
            auto host_keys = get_host_access(keys_first);
            auto host_vals = get_host_access(vals_first);
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);

            initialize_data(host_keys, host_vals, host_key_res, host_val_res, n);
        }

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::reduce_by_segment(new_policy3, keys_first, keys_last, vals_first, key_res_first,
                                                   val_res_first, [](KeyT first, KeyT second) { return first == second; },
                                                   [](ValT first, ValT second) { return first + second; });
        new_policy3.queue().wait_and_throw();
        result_size = std::distance(key_res_first, res3.first);

        {
            auto host_key_res = get_host_access(key_res_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_key_res, host_val_res, result_size);
        }
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif
            is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        initialize_data(keys_first, vals_first, key_res_first, val_res_first, n);

        auto res1 =
            oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first, val_res_first);
        Size result_size = std::distance(key_res_first, res1.first);
        check_values(key_res_first, val_res_first, result_size);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, key_res_first, val_res_first, n);
        auto res2 = oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first,
                                                   val_res_first, [](KeyT first, KeyT second) { return first == second; });
        result_size = std::distance(key_res_first, res2.first);
        check_values(key_res_first, val_res_first, result_size);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, key_res_first, val_res_first, n);
        auto res3 = oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first,
                                                   val_res_first, [](KeyT first, KeyT second) { return first == second; },
                                                   [](ValT first, ValT second) { return first + second; });
        result_size = std::distance(key_res_first, res3.first);
        check_values(key_res_first, val_res_first, result_size);
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    typename ::std::enable_if<!is_same_iterator_category<Iterator3, ::std::random_access_iterator_tag>::value,
                              void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
    }
};

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test4buffers<char, uint64_t, test_reduce_by_segment>();
#endif
    test_algo_four_sequences<uint64_t, int, test_reduce_by_segment>();
    return done();
}
