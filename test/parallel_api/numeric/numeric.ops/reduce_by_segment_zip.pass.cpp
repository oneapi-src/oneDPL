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
#include <vector>

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q;

    const int n = 9, n_res = 6;

    std::vector<int> keys1{11, 11, 21, 20, 21, 21, 21, 37, 37};
    std::vector<int> keys2{11, 11, 20, 20, 20, 21, 21, 37, 37};
    std::vector<int> values{0,  1,  2,  3,  4,  5,  6,  7,  8};
    std::vector<int> output_keys1(keys1.size());
    std::vector<int> output_keys2(keys2.size());    
    std::vector<int> output_values(values.size());

    int* d_keys1         = sycl::malloc_device<int>(n, q);
    int* d_keys2         = sycl::malloc_device<int>(n, q);
    int* d_values        = sycl::malloc_device<int>(n, q);
    int* d_output_keys1  = sycl::malloc_device<int>(n, q);
    int* d_output_keys2  = sycl::malloc_device<int>(n, q);
    int* d_output_values = sycl::malloc_device<int>(n, q);

    q.memcpy(d_keys1, keys1.data(), sizeof(int)*n);
    q.memcpy(d_keys2, keys2.data(), sizeof(int)*n);
    q.memcpy(d_values, values.data(), sizeof(int)*n);

    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);

    auto new_last = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::make_device_policy(q),
        begin_keys_in, end_keys_in, d_values, begin_keys_out, d_output_values);

    q.memcpy(output_keys1.data(), d_output_keys1, sizeof(int)*n);
    q.memcpy(output_keys2.data(), d_output_keys2, sizeof(int)*n);    
    q.memcpy(output_values.data(), d_output_values, sizeof(int)*n);
    q.wait();

    // Expected output
    // {11, 11}: 1
    // {21, 20}: 2
    // {20, 20}: 3
    // {21, 20}: 4
    // {21, 21}: 11
    // {37, 37}: 15
    for(int i=0; i < n_res; i++) {
      std::cout << "{" << output_keys1[i] << ", " << output_keys2[i] << "}: " << output_values[i] << std::endl;
    }

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
