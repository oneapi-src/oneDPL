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

template <sycl::usm::alloc alloc_type, typename KernelName>
void
test_with_usm()
{
    sycl::queue q = TestUtils::get_test_queue();

    constexpr int n = 9;

    //data initialization
    int keys1 [n] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
    int keys2 [n] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
    int values[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int output_keys1 [n] = { };
    int output_keys2 [n] = { };
    int output_values[n] = { };

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper1(q, std::begin(keys1),         std::end(keys1));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper2(q, std::begin(keys2),         std::end(keys2));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper3(q, std::begin(values),        std::end(values));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper4(q, std::begin(output_keys1),  std::end(output_keys1));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper5(q, std::begin(output_keys2),  std::end(output_keys2));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper6(q, std::begin(output_values), std::end(output_values));
    auto d_keys1         = dt_helper1.get_data();
    auto d_keys2         = dt_helper2.get_data();
    auto d_values        = dt_helper3.get_data();
    auto d_output_keys1  = dt_helper4.get_data();
    auto d_output_keys2  = dt_helper5.get_data();
    auto d_output_values = dt_helper6.get_data();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);

    //run reduce_by_segment algorithm 
    auto new_last = oneapi::dpl::reduce_by_segment(
        TestUtils::make_device_policy<KernelName>(q), begin_keys_in,
        end_keys_in, d_values, begin_keys_out, d_output_values);

    q.wait();

    //retrieve result on the host and check the result
    dt_helper4.retrieve_data(output_keys1);
    dt_helper5.retrieve_data(output_keys2);
    dt_helper6.retrieve_data(output_values);

//Dump
#if 0
    for(int i=0; i < n; i++) {
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
    const int exp_keys1[n] = {11, 21, 20, 21, 21,37};
    const int exp_keys2[n] = {11, 20, 20, 20, 21, 37};
    const int exp_values[n] = {1, 2, 3, 4, 11, 15};
    EXPECT_EQ_N(exp_keys1, output_keys1, n, "wrong keys1 from reduce_by_segment");
    EXPECT_EQ_N(exp_keys2, output_keys2, n, "wrong keys2 from reduce_by_segment");
    EXPECT_EQ_N(exp_values, output_values, n, "wrong values from reduce_by_segment");
}
#endif

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared, class KernelName1>();
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device, class KernelName2>();
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
