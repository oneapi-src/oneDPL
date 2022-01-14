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

//#define DUMP_CHECK_RESULTS

template <typename BinaryOperation>
struct test_inclusive_scan_by_segment
{
    template <typename T>
    struct DifferenceTypeT
    {
        using DifferenceType = typename ::std::iterator_traits<T>::difference_type;
    };

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename T, int Dim, sycl::access::mode AccMode, sycl::access::target AccTarget, sycl::access::placeholder Placeholder>
    struct DifferenceTypeT<sycl::accessor<T, Dim, AccMode, AccTarget, Placeholder>>
    {
        using DifferenceType = ::std::size_t;
    };
#endif // TEST_DPCPP_BACKEND_PRESENT

    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, ... };

        using DifferenceType = typename DifferenceTypeT<decltype(host_vals)>::DifferenceType;

        DifferenceType segment_length = 1;
        for (DifferenceType i = 0; i != n; )
        {
          for (DifferenceType j = 0; j != 4*segment_length && i != n; ++j)
          {
              host_keys[i] = j/segment_length + 1;
              host_vals[i] = 1;
              host_val_res[i] = 0;
              ++i;
          }
          ++segment_length;
        }
    }

//#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        using DifferenceType = typename DifferenceTypeT<Iterator>::DifferenceType;

        std::cout << msg;
        for (DifferenceType i = 0; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;
    }
//#endif // DUMP_CHECK_RESULTS

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    check_values(Iterator1 host_keys, Iterator2 host_vals, Iterator3 val_res, Size n)
    {
        using ValT = typename ::std::decay<decltype(host_vals[0])>::type;

        return check_values(host_keys, host_vals, val_res, n, ::std::plus<ValT>());
    }

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename BinaryOperationCheck>
    void
    check_values(Iterator1 host_keys, Iterator2 host_vals, Iterator3 val_res, Size n, BinaryOperationCheck op)
    {
        // https://docs.oneapi.io/versions/latest/onedpl/extension_api.html
        // keys:   [ 0, 0, 0, 1, 1, 1 ]
        // values: [ 1, 2, 3, 4, 5, 6 ]
        // result: [ 1, 1 + 2 = 3, 1 + 2 + 3 = 6, 4, 4 + 5 = 9, 4 + 5 + 6 = 15 ]

#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << n << ") : " << std::endl;
        display_param("keys:   ", host_keys, n);
        display_param("values: ", host_vals, n);
        display_param("result: ", val_res,   n);
#endif // DUMP_CHECK_RESULTS

        if (n < 1)
            return;

        if (n == 1)
        {
            EXPECT_TRUE(host_vals[0] == val_res[0], "wrong effect from exclusive_scan_by_segment");
            return;
        }

        auto init = host_vals[0];
        init = {};

        using DifferenceType = typename DifferenceTypeT<decltype(host_vals)>::DifferenceType;

        // Last summ info
        DifferenceType last_segment_begin = 0;  // Start index of last summ
        DifferenceType last_segment_end = 0;    // End index of last summ
        auto last_segment_summ = init;          // Last summ

        DifferenceType segment_start_idx = 0;
        DifferenceType val_res_idx = 0;
        for (DifferenceType current_key_idx = 1; current_key_idx <= n; ++current_key_idx)
        {
            // Eval current summ
            auto expected_segment_sum = init;
            if (last_segment_begin == segment_start_idx && last_segment_end + 1 == current_key_idx)
            {
                expected_segment_sum = segment_start_idx == current_key_idx - 1
                                           ? host_vals[segment_start_idx]
                                           : op(last_segment_summ, host_vals[current_key_idx - 1]);
            }
            else
            {
                assert(segment_start_idx == current_key_idx - 1);
                expected_segment_sum = host_vals[segment_start_idx];
            }

            // Update last summ  info
            last_segment_begin = segment_start_idx;
            last_segment_end = current_key_idx;
            last_segment_summ = expected_segment_sum;

            if (val_res[val_res_idx] != expected_segment_sum)
            {
                std::cout << "check_values(n = " << n << ") : val_res[val_res_idx] != expected_segment_sum : "
                    << "val_res_idx = " << val_res_idx << ";"
                    << "val_res[val_res_idx]" << val_res[val_res_idx] << ";"
                    << "expected_segment_sum = " << expected_segment_sum
                    << std::endl;

                display_param("keys:   ", host_keys, n);
                display_param("values: ", host_vals, n);
                display_param("result: ", val_res, n);
            }

            EXPECT_TRUE(val_res[val_res_idx] == expected_segment_sum, "wrong effect from exclusive_scan_by_segment");
            ++val_res_idx;

            if (current_key_idx < n && host_keys[segment_start_idx] != host_keys[current_key_idx])
                segment_start_idx = current_key_idx;
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
            auto host_vals = get_host_access(vals_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_keys, host_vals, host_val_res, n);

            // call algorithm with equality comparator

            initialize_data(host_keys, host_vals, host_val_res, n);
        }

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(new_policy2, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; });
        exec.queue().wait_and_throw();
        {
            auto host_keys = get_host_access(keys_first);
            auto host_vals = get_host_access(vals_first);
            auto host_val_res = get_host_access(val_res_first);
            check_values(host_keys, host_vals, host_val_res, n);

            // call algorithm with addition operator

            initialize_data(host_keys, host_vals, host_val_res, n);
        }

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(new_policy3, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           BinaryOperation());
        exec.queue().wait_and_throw();
        auto host_keys = get_host_access(keys_first);
        auto host_vals = get_host_access(vals_first);
        auto host_val_res = get_host_access(val_res_first);
        check_values(host_keys, host_vals, host_val_res, n, BinaryOperation());
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

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res1 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first);
        check_values(keys_first, vals_first, val_res_first, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res2 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; });
        check_values(keys_first, vals_first, val_res_first, n);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::inclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           BinaryOperation());
        check_values(keys_first, vals_first, val_res_first, n, BinaryOperation());
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

template<typename _Tp>
struct UserBinaryOperation
{
    _Tp operator()(const _Tp& __x, const _Tp& __y) const
    {
        return __x * __y;
    }
};

int main()
{
    std::cout << "Test operation: ::std::plus<ValueType>" << std::endl;
    {
        using ValueType = ::std::uint64_t;
        using BinaryOperation = ::std::plus<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        test3buffers<ValueType, test_inclusive_scan_by_segment<BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<ValueType, test_inclusive_scan_by_segment<BinaryOperation>>();
    }

    std::cout << "Test operation: UserBinaryOperation (op1 * op2)" << std::endl;
    {
        using ValueType = ::std::int64_t;
        using BinaryOperation = UserBinaryOperation<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        test3buffers<ValueType, test_inclusive_scan_by_segment<BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<ValueType, test_inclusive_scan_by_segment<BinaryOperation>>();
    }
    return TestUtils::done();
}
