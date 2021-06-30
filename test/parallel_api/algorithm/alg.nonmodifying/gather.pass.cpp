// -*- C++ -*-
//===-- gather.pass.cpp ----------------------------------------------------===//
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

// Tests for gather

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#include <random> //std::default_random_engine

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
OutputIter
gather(Policy&& policy, InputIter1 map_first, InputIter1 map_last, InputIter2 input_first, OutputIter result)
{
    auto perm_begin = oneapi::dpl::make_permutation_iterator(input_first, map_first);
    const int n = ::std::distance(map_first, map_last);

    return oneapi::dpl::copy(policy, perm_begin, perm_begin + n, result);
}

void
test_gather_with_usm(int input_size)
{
    sycl::queue q;

    auto sycl_deleter = [q](int* mem) { sycl::free(mem, q.get_context()); };

    ::std::unique_ptr<int, decltype(sycl_deleter)>

        data((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

        indices((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter),

        result((int*)sycl::malloc_shared(sizeof(int) * input_size, q.get_device(), q.get_context()), sycl_deleter);

    int* data_ptr = data.get();
    int* idx_ptr = indices.get();
    int* res_ptr = result.get();

    for (int i = 0; i != input_size; ++i)
    {
        data_ptr[i] = i + 1; // data = {1, 2, 3, ..., n}
        idx_ptr[i] = i;      // indices = {0, 1, 2, ..., n-1}
        res_ptr[i] = 0;      // result = {0, 0, 0, ..., 0}
    }

    // obtain a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // randomize map indices
    shuffle(idx_ptr, idx_ptr + input_size, std::default_random_engine(seed));

    // call gather
    gather(oneapi::dpl::execution::dpcpp_default, idx_ptr, idx_ptr + input_size, data_ptr, res_ptr);

    q.wait_and_throw();

    // test if sum of input and result arrays are equal
    if (std::accumulate(data_ptr, data_ptr + input_size, 0) != std::accumulate(res_ptr, res_ptr + input_size, 0))
    {
        std::cout << "Input size " << input_size << ": Failed\n";
    }
}

void
test_gather_with_buffers(int input_size)
{
    sycl::buffer<uint64_t, 1> src_buf{sycl::range<1>(input_size)};
    sycl::buffer<uint64_t, 1> map_buf{sycl::range<1>(input_size)};
    sycl::buffer<uint64_t, 1> res_buf{sycl::range<1>(input_size)};

    {
        auto src = src_buf.template get_access<sycl::access::mode::write>();
        auto map = map_buf.template get_access<sycl::access::mode::write>();
        auto res = res_buf.template get_access<sycl::access::mode::write>();

        for (int i = 0; i != input_size; ++i)
        {
            src[i] = i + 1;
            map[i] = input_size - i - 1;
            res[i] = 0;
        }
    }

    // create sycl iterators
    auto src_beg = oneapi::dpl::begin(src_buf);
    auto map_beg = oneapi::dpl::begin(map_buf);
    auto map_end = oneapi::dpl::end(map_buf);
    auto res_beg = oneapi::dpl::begin(res_buf);

    // call algorithm
    gather(oneapi::dpl::execution::dpcpp_default, map_beg, map_end, src_beg, res_beg);

    auto source = src_buf.template get_access<sycl::access::mode::read>();
    auto result = res_buf.template get_access<sycl::access::mode::read>();

    int src_sum, res_sum = 0;

    // test if sum of input and result arrays are equal
    for (int i = 0; i != input_size; ++i)
    {
        src_sum += source[i];
        res_sum += result[i];
    }
    if (src_sum != res_sum)
    {
        std::cout << "Input size " << input_size << ": Failed\n";
    }
}

int
main()
{
    const int max_n = 100000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_gather_with_usm(n);
        test_gather_with_buffers(n);
    }

    return TestUtils::done();
}
