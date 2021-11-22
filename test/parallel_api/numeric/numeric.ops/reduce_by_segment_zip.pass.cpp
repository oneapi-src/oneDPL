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

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>
#endif

#include "support/utils.h"

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(functional)

#include <functional>
#include <iostream>

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q;

    if(!TestUtils::has_aspect(q.get_device(), sycl::aspect::usm_shared_allocations))
        return TestUtils::done(0);

    const int n = 9, n_res = 6;

    //shared memory allocation
    int* d_keys1         = sycl::malloc_shared<int>(n, q);
    int* d_keys2         = sycl::malloc_shared<int>(n, q);
    int* d_values        = sycl::malloc_shared<int>(n, q);
    int* d_output_keys1  = sycl::malloc_shared<int>(n, q);
    int* d_output_keys2  = sycl::malloc_shared<int>(n, q);
    int* d_output_values = sycl::malloc_shared<int>(n, q);

    //data initialization
    const int keys1[n] = {11, 11, 21, 20, 21, 21, 21, 37, 37};
    const int keys2[n] = {11, 11, 20, 20, 20, 21, 21, 37, 37};
    const int values[n] = {0,  1,  2,  3,  4,  5,  6,  7,  8};
    std::copy(keys1, keys1 + n, d_keys1);
    std::copy(keys2, keys2 + n, d_keys2);
    std::copy(values, values + n, d_values);

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);

    //run reduce_by_segment algorithm
    auto new_last = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q),
        begin_keys_in, end_keys_in, d_values, begin_keys_out, d_output_values);

    q.wait();

//Dump
#if 0
    for(int i=0; i < n_res; i++) {
      std::cout << "{" << d_output_keys1[i] << ", " << d_output_keys2[i] << "}: " << d_output_values[i] << std::endl;
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
    EXPECT_EQ_N(exp_keys1, d_output_keys1, n_res, "wrong keys1 from reduce_by_segment");
    EXPECT_EQ_N(exp_keys2, d_output_keys2, n_res, "wrong keys2 from reduce_by_segment");
    EXPECT_EQ_N(exp_values, d_output_values, n_res, "wrong values from reduce_by_segment");

    // Deallocate memory
    sycl::free(d_keys1, q);
    sycl::free(d_keys2, q);
    sycl::free(d_values, q);
    sycl::free(d_output_keys1, q);
    sycl::free(d_output_keys2, q);
    sycl::free(d_output_values, q);

#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
