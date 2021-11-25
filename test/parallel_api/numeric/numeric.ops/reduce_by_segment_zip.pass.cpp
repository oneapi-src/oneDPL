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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(functional)

#include <functional>
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type>
void
test_with_usm()
{
    sycl::queue q;

    //data initialization
    auto prepare_data = [](int n, int* keys1, int* keys2, int* values)
        {
            constexpr int items_count = 9;

            const int src_keys1 [items_count] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
            const int src_keys2 [items_count] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
            const int src_values[items_count] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };

            ::std::copy_n(src_keys1, items_count, keys1);
            ::std::copy_n(src_keys2, items_count, keys2);
            ::std::copy_n(src_values, items_count, values);
        };

    constexpr int n = 9;
    constexpr int n_res = 6;
    int keys1[n] = {};
    int keys2[n] = {};
    int values[n] = {};
    int output_keys1[n] = {};
    int output_keys2[n] = {};
    int output_values[n] = {};

    prepare_data(n, keys1, keys2, values);

    // allocate USM memory and copying data to USM shared/device memory
    auto usmPtr1 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, keys1,         n);
    auto usmPtr2 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, keys2,         n);
    auto usmPtr3 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, values,        n);
    auto usmPtr4 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, output_keys1,  n);
    auto usmPtr5 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, output_keys2,  n);
    auto usmPtr6 = TestUtils::usm_alloc_and_copy<alloc_type, int>(q, output_values, n);
    auto d_keys1         = usmPtr1.get();
    auto d_keys2         = usmPtr2.get();
    auto d_values        = usmPtr3.get();
    auto d_output_keys1  = usmPtr4.get();
    auto d_output_keys2  = usmPtr5.get();
    auto d_output_values = usmPtr6.get();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);

    //run reduce_by_segment algorithm 
    auto new_last = oneapi::dpl::reduce_by_segment(
        oneapi::dpl::execution::make_device_policy(q), begin_keys_in,
        end_keys_in, d_values, begin_keys_out, d_output_values);

    q.wait();

    //retrieve result on the host and check the result
    TestUtils::retrieve_data(q, usmPtr4, output_keys1,  n);
    TestUtils::retrieve_data(q, usmPtr5, output_keys2,  n);
    TestUtils::retrieve_data(q, usmPtr6, output_values, n);

//Dump
#if 0
    for(int i=0; i < n_res; i++) {
      std::cout << "{" << output_keys1[i] << ", " << output_keys2[i] << "}: " << output_values[i] << std::endl;
    }
#endif

    // Expected output
    // {11, 11}: 1
    // {21, 20}: 2
    // {20, 20}: 3
    // {21, 20}: 4
    // {21, 21}: 11
    // {37, 37}: 15
    const int exp_keys1[n_res] = {11, 21, 20, 21, 21,37};
    const int exp_keys2[n_res] = {11, 20, 20, 20, 21, 37};
    const int exp_values[n_res] = {1, 2, 3, 4, 11, 15};
    EXPECT_EQ_N(exp_keys1, output_keys1, n_res, "wrong keys1 from reduce_by_segment");
    EXPECT_EQ_N(exp_keys2, output_keys2, n_res, "wrong keys2 from reduce_by_segment");
    EXPECT_EQ_N(exp_values, output_values, n_res, "wrong values from reduce_by_segment");
}
#endif

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared>();
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device>();
#endif

    return TestUtils::done();
}
