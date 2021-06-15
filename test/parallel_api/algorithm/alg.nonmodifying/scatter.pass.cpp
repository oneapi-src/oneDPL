// -*- C++ -*-
//===-- scatter.pass.cpp ----------------------------------------------------===//
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

// Tests for scatter

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#include <random> // std::default_random_engine

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
void scatter(Policy&& policy, InputIter1 first, InputIter1 last, InputIter2 map, OutputIter result) {
    oneapi::dpl::copy(policy, first, last, oneapi::dpl::make_permutation_iterator(result, map));
}

void test_scatter(int input_size) {
    sycl::queue q;

    auto sycl_deleter = [q](int* mem) { sycl::free(mem, q.get_context()); };

    ::std::unique_ptr<int, decltype(sycl_deleter)>

    data((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    indices((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

    result((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter);

    int* data_ptr = data.get(); 
    int* idx_ptr = indices.get(); 
    int* res_ptr = result.get();

    for (int i = 0; i != input_size; ++i) {
        data_ptr[i] = i+1; // data = {1, 2, 3, ..., n}
        idx_ptr[i] = i; // indices = {0, 1, 2, ..., n-1} 
        res_ptr[i] = 0; // result = {0, 0, 0, ..., 0}
    }   

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // randomize map indices
    shuffle(idx_ptr, idx_ptr + input_size, std::default_random_engine(seed));

    // call scatter
    scatter(oneapi::dpl::execution::dpcpp_default, data_ptr, data_ptr + input_size, idx_ptr, res_ptr);

    q.wait_and_throw();

    // test if sum of input and result arrays are equal
    if (std::accumulate(data_ptr, data_ptr + input_size, 0) != std::accumulate(res_ptr, res_ptr + input_size, 0)) {
        std::cout << "Input size " << input_size << ": Failed\n";
    }
}

int main() {
    const int max_n = 100000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        test_scatter(n);
    }

    return TestUtils::done();
}
