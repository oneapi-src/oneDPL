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
    using SyclHelper = TestUtils::sycl_operations_helper<alloc_type, int>;

    sycl::queue q;

    constexpr int n = 9;
    constexpr int n_res = 6;

    //shared memory allocation
    auto d_keys1         = SyclHelper::alloc_ptr(q, n);
    auto d_keys2         = SyclHelper::alloc_ptr(q, n);
    auto d_values        = SyclHelper::alloc_ptr(q, n);
    auto d_output_keys1  = SyclHelper::alloc_ptr(q, n);
    auto d_output_keys2  = SyclHelper::alloc_ptr(q, n);
    auto d_output_values = SyclHelper::alloc_ptr(q, n);

    //data initialization
    const int keys1 [n] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
    const int keys2 [n] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
    const int values[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    std::copy(keys1, keys1 + n, d_keys1.get());
    std::copy(keys2, keys2 + n, d_keys2.get());
    std::copy(values, values + n, d_values.get());

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1.get(), d_keys2.get());
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1.get() + n, d_keys2.get() + n);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1.get(), d_output_keys2.get());

    //run reduce_by_segment algorithm 
    auto new_last = oneapi::dpl::reduce_by_segment(
        oneapi::dpl::execution::make_device_policy(q), begin_keys_in,
        end_keys_in, d_values.get(), begin_keys_out, d_output_values.get());

    q.wait();

    int d_output_keys1_on_host [n] = { };
    int d_output_keys2_on_host [n] = { };
    int d_output_values_on_host[n] = { };

    SyclHelper::copy_to_host(q, d_output_keys1_on_host,  d_output_keys1.get(),  n);
    SyclHelper::copy_to_host(q, d_output_keys2_on_host,  d_output_keys2.get(),  n);
    SyclHelper::copy_to_host(q, d_output_values_on_host, d_output_values.get(), n);

//Dump
#if 0
    for(int i=0; i < n_res; i++) {
      std::cout << "{" << d_output_keys1_on_host[i] << ", " << d_output_keys2_on_host[i] << "}: " << d_output_values_on_host[i] << std::endl;
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
    EXPECT_EQ_N(exp_keys1, d_output_keys1_on_host, n_res, "wrong keys1 from reduce_by_segment");
    EXPECT_EQ_N(exp_keys2, d_output_keys2_on_host, n_res, "wrong keys2 from reduce_by_segment");
    EXPECT_EQ_N(exp_values, d_output_values_on_host, n_res, "wrong values from reduce_by_segment");
}
#endif

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_usm<sycl::usm::alloc::shared>();
    test_with_usm<sycl::usm::alloc::device>();
#endif

    return TestUtils::done();
}
