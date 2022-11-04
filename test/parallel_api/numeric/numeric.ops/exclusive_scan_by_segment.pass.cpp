// -*- C++ -*-
//===-- exclusive_scan_by_segment.pass.cpp ------------------------------------===//
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
#include "support/scan_serial_impl.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"

using namespace oneapi::dpl::execution;
#endif // TEST_DPCPP_BACKEND_PRESENT
using namespace TestUtils;

// This macro may be used to analyze source data and test results in test_inclusive_scan_by_segment
// WARNING: in the case of using this macro debug output is very large.
//#define DUMP_CHECK_RESULTS

DEFINE_TEST_1(test_exclusive_scan_by_segment, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_exclusive_scan_by_segment)

    // TODO: replace data generation with random data and update check to compare result to
    // the result of a serial implementation of the algorithm
    template <typename Accessor1, typename Accessor2, typename Accessor3, typename Size>
    void initialize_data(Accessor1 host_keys, Accessor2 host_vals, Accessor3 host_val_res, Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, ... };

        int segment_length = 1;
        int i = 0;
        while (i != n)
        {
            for (int j = 0; j != 4 * segment_length && i != n; ++j)
            {
                host_keys[i] = j / segment_length + 1;
                host_vals[i] = 1;
                host_val_res[i] = 0;
                ++i;
            }
            ++segment_length;
        }
    }

#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        std::cout << msg;
        for (Size i = 8000; i < n; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << it[i];
        }
        std::cout << std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    template <typename Accessor1, typename Accessor2, typename Accessor3, typename T, typename Size,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(Accessor1 host_keys, Accessor2 host_vals, Accessor3 val_res, T init, Size n,
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n1] = { 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, ...};

#ifdef DUMP_CHECK_RESULTS
        std::cout << "check_values(n = " << n << ") : " << std::endl;
        display_param("         keys:   ", host_keys, n);
        display_param("         values: ", host_vals, n);
        display_param("         result: ", val_res, n);
#endif // DUMP_CHECK_RESULTS

        assert(init == 0 || init == 1);
        int segment_length = 1;
        auto expected_segment_sum = init;
        using key_type = typename std::decay_t<decltype(host_keys[0])>;
        key_type current_key = key_type(999); //not one of the input keys

        using value_type = typename ::std::decay_t<decltype(val_res[0])>;

        std::vector<value_type> expected_val_res(n);
        exclusive_scan_by_segment_serial(host_keys, host_vals, expected_val_res, n, init, op);
#ifdef DUMP_CHECK_RESULTS
        display_param("expected result: ", expected_val_res.data(), n);
#endif // DUMP_CHECK_RESULTS
        for (int i = 0; i != n; ++i)
        {
            if (val_res[i] != expected_val_res[i])
                std::cout << val_res[i] << " " << expected_val_res[i] << " " << i << std::endl;
            EXPECT_TRUE(val_res[i] == expected_val_res[i], "wrong effect from exclusive_scan_by_segment");
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
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes, Size> host_val_res(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;
        typedef typename ::std::iterator_traits<Iterator3>::value_type ValR;

        const ValT init = 1;

        // call algorithm with no optional arguments
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy, keys_first, keys_last, vals_first, val_res_first, init);
        exec.queue().wait_and_throw();

        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), init, n);

        // call algorithm with equality comparator
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy2, keys_first, keys_last, vals_first, val_res_first, init,
                                                   [](KeyT first, KeyT second) { return first == second; });
        exec.queue().wait_and_throw();

        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), init, n);

        // call algorithm with addition operator
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto binary_op = [](ValT first, ValT second) { return first + second; };
        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy3, keys_first, keys_last, vals_first, val_res_first, init,
                                                   [](KeyT first, KeyT second) { return first == second; }, binary_op);
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), init, n, binary_op);

        auto new_policy4 = make_new_policy<new_kernel_name<Policy, 3>>(exec);
        auto res4 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy4, keys_first, keys_last, vals_first, val_res_first);
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), 0, n);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    typename ::std::enable_if<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy<typename ::std::decay<Policy>::type>::value &&
#endif // TEST_DPCPP_BACKEND_PRESENT
            is_base_of_iterator_category<::std::random_access_iterator_tag, Iterator3>::value,
        void>::type
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        const ValT init = 1;
        const ValT zero = 0;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res1 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first);
        check_values(keys_first, vals_first, val_res_first, zero, n);

        // call algorithm with initial value
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res2 =
            oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, init);
        check_values(keys_first, vals_first, val_res_first, init, n);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, zero,
                                                           [](KeyT first, KeyT second) { return first == second; });
        check_values(keys_first, vals_first, val_res_first, zero, n);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res4 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, init,
                                                           [](KeyT first, KeyT second) { return first == second; },
                                                           BinaryOperation());
        check_values(keys_first, vals_first, val_res_first, init, n, BinaryOperation());
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

template <typename _Tp>
struct UserBinaryOperation
{
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        return __x * __y;
    }
};

int
main()
{
    {
        using ValueType = std::uint64_t;
        using BinaryOperation = ::std::plus<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        // Run tests for USM shared memory
        test3buffers<sycl::usm::alloc::shared, test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
        // Run tests for USM device memory
        test3buffers<sycl::usm::alloc::device, test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
#else
        test_algo_three_sequences<ValueType, test_exclusive_scan_by_segment>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
    {
        using ValueType = std::int64_t;
        using BinaryOperation = UserBinaryOperation<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        // Run tests for USM shared memory
        test3buffers<sycl::usm::alloc::shared, test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
        // Run tests for USM device memory
        test3buffers<sycl::usm::alloc::device, test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<test_exclusive_scan_by_segment<ValueType, BinaryOperation>>();
#else
        test_algo_three_sequences<ValueType, test_exclusive_scan_by_segment>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

    return TestUtils::done();
}
