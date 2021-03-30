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

#include <iostream>
#include <iomanip>

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (EXCLUSIVE_SCAN_BY_SEGMENT): fail (" << X << "," << Y << ")" << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>

void test_with_buffers()
{
    // create a buffer, being responsible for moving data around and counting dependencies
    cl::sycl::buffer<uint64_t, 1> _key_buf{ cl::sycl::range<1>(10) };
    cl::sycl::buffer<uint64_t, 1> _val_buf{ cl::sycl::range<1>(10) };
    cl::sycl::buffer<uint64_t, 1> _res_buf{ cl::sycl::range<1>(10) };

    {
        auto key_buf = _key_buf.template get_access<cl::sycl::access::mode::read_write>();
        auto val_buf = _val_buf.template get_access<cl::sycl::access::mode::read_write>();
        auto res_buf = _res_buf.template get_access<cl::sycl::access::mode::read_write>();

        // Initialize data
        key_buf[0] = 0; key_buf[1] = 0; key_buf[2] = 0; key_buf[3] = 1; key_buf[4] = 1;
        key_buf[5] = 2; key_buf[6] = 3; key_buf[7] = 3; key_buf[8] = 3; key_buf[9] = 3;

        val_buf[0] = 1; val_buf[1] = 1; val_buf[2] = 1; val_buf[3] = 1; val_buf[4] = 1;
        val_buf[5] = 1; val_buf[6] = 1; val_buf[7] = 1; val_buf[8] = 1; val_buf[9] = 1;

        res_buf[0] = 9; res_buf[1] = 9; res_buf[2] = 9; res_buf[3] = 9; res_buf[4] = 9;
        res_buf[5] = 9; res_buf[6] = 9; res_buf[7] = 9; res_buf[8] = 9; res_buf[9] = 9;
    }

    // create sycl iterators
    auto key_beg = oneapi::dpl::begin(_key_buf);
    auto key_end = oneapi::dpl::end(_key_buf);
    auto val_beg = oneapi::dpl::begin(_val_buf);
    auto res_beg = oneapi::dpl::begin(_res_buf);

    // create named policy from existing one
    auto new_policy = oneapi::dpl::execution::make_device_policy<class ExclusiveScanBySegment>(oneapi::dpl::execution::dpcpp_default);
    // call algorithm
    oneapi::dpl::exclusive_scan_by_segment(new_policy, key_beg, key_end, val_beg, res_beg,
        (uint64_t)0, std::equal_to<uint64_t>(), std::plus<uint64_t>());

    // check values
    auto key_acc = _key_buf.get_access<cl::sycl::access::mode::read>();
    auto val_acc = _val_buf.get_access<cl::sycl::access::mode::read>();
    auto res_acc = _res_buf.get_access<cl::sycl::access::mode::read>();
    uint64_t check_value;
    for (int i = 0; i != 10; ++i) {
            if (i == 0 || key_acc[i] != key_acc[i-1])
                check_value = 0;
        else
        check_value += val_acc[i-1];
        ASSERT_EQUAL(check_value, res_acc[i]);
    }
}

void test_with_usm()
{
    cl::sycl::queue q;
    const int n = 10;

    // Allocate space for data using USM.
    uint64_t* key_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
    uint64_t* val_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
    uint64_t* res_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));

    // Initialize data
    key_head[0] = 0; key_head[1] = 0; key_head[2] = 0; key_head[3] = 1; key_head[4] = 1;
    key_head[5] = 2; key_head[6] = 3; key_head[7] = 3; key_head[8] = 3; key_head[9] = 3;

    val_head[0] = 1; val_head[1] = 1; val_head[2] = 1; val_head[3] = 1; val_head[4] = 1;
    val_head[5] = 1; val_head[6] = 1; val_head[7] = 1; val_head[8] = 1; val_head[9] = 1;

    res_head[0] = 9; res_head[1] = 9; res_head[2] = 9; res_head[3] = 9; res_head[4] = 9;
    res_head[5] = 9; res_head[6] = 9; res_head[7] = 9; res_head[8] = 9; res_head[9] = 9;

    // call algorithm
    auto new_policy = oneapi::dpl::execution::make_device_policy<class exclusive_scan_by_segment_1>(q);
    oneapi::dpl::exclusive_scan_by_segment(new_policy, key_head, key_head+n, val_head, res_head,
        (uint64_t)0, std::equal_to<uint64_t>(), std::plus<uint64_t>());
    q.wait();

    // check values
    uint64_t check_value;
    for (int i = 0; i != 10; ++i) {
        if (i == 0 || key_head[i] != key_head[i-1])
            check_value = 0;
        else
    	check_value += val_head[i-1];
        ASSERT_EQUAL(check_value, res_head[i]);
    }

    // call algorithm on single element range
    res_head[0] = 9;
    auto new_policy2 = oneapi::dpl::execution::make_device_policy<class exclusive_scan_by_segment_2>(q);
    oneapi::dpl::exclusive_scan_by_segment(new_policy2, key_head, key_head+1, val_head, res_head,
        (uint64_t)0);

    // check values
    ASSERT_EQUAL(0, res_head[0]);
}
#endif

void test_on_host() {
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int init = 5;

    oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result, init, std::equal_to<int>(), std::plus<int>());

    // data is now {5,6,7,5,6,5,5,6,7,8};
    ASSERT_EQUAL(result[0], 5);
    ASSERT_EQUAL(result[1], 6);
    ASSERT_EQUAL(result[2], 7);
    ASSERT_EQUAL(result[3], 5);
    ASSERT_EQUAL(result[4], 6);
    ASSERT_EQUAL(result[5], 5);
    ASSERT_EQUAL(result[6], 5);
    ASSERT_EQUAL(result[7], 6);
    ASSERT_EQUAL(result[8], 7);
    ASSERT_EQUAL(result[9], 8);
}

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
    test_with_usm();
#endif
    test_on_host();

    return TestUtils::done();
}
