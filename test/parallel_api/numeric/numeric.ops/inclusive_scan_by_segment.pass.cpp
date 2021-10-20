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

#include <iostream>
#include <iomanip>

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (INCLUSIVE_SCAN_BY_SEGMENT): fail (" << X << "," << Y << ")" << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

void test_with_buffers()
{
    // create a buffer, being responsible for moving data around and counting dependencies
    sycl::buffer<uint64_t, 1> _key_buf{ sycl::range<1>(10) };
    sycl::buffer<uint64_t, 1> _val_buf{ sycl::range<1>(10) };
    sycl::buffer<uint64_t, 1> _res_buf{ sycl::range<1>(10) };

    {
    auto key_buf = _key_buf.template get_access<sycl::access::mode::read_write>();
    auto val_buf = _val_buf.template get_access<sycl::access::mode::read_write>();
    auto res_buf = _res_buf.template get_access<sycl::access::mode::read_write>();

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
    auto new_policy = oneapi::dpl::execution::make_device_policy<class InclusiveScanBySegment>(oneapi::dpl::execution::dpcpp_default);
    // call algorithm
    oneapi::dpl::inclusive_scan_by_segment(new_policy, key_beg, key_end, val_beg, res_beg,
        std::equal_to<uint64_t>(), std::plus<uint64_t>());

    // check values
    auto key_acc = _key_buf.get_access<sycl::access::mode::read>();
    auto val_acc = _val_buf.get_access<sycl::access::mode::read>();
    auto res_acc = _res_buf.get_access<sycl::access::mode::read>();
    uint64_t check_value;
    for (int i = 0; i != 10; ++i) {
        if (i == 0 || key_acc[i] != key_acc[i-1])
            check_value = val_acc[i];
        else
            check_value += val_acc[i];
        ASSERT_EQUAL(check_value, res_acc[i]);
    }
}

template <sycl::usm::alloc alloc_type>
void
test_with_usm()
{
    using SyclHelper = TestUtils::sycl_operations_helper<alloc_type, uint64_t>;

    sycl::queue q;
    constexpr int n = 10;

    // Initialize data
    uint64_t key_head_on_host[n] = { 0, 0, 0, 1, 1, 2, 3, 3, 3, 3 };
    uint64_t val_head_on_host[n] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    uint64_t res_head_on_host[n] = { 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };

    // Allocate space for data using USM.
    uint64_t* key_head = SyclHelper::alloc(q, n);
    uint64_t* val_head = SyclHelper::alloc(q, n);
    uint64_t* res_head = SyclHelper::alloc(q, n);

    SyclHelper::cpy_from_host(q, key_head, key_head_on_host, n);
    SyclHelper::cpy_from_host(q, val_head, val_head_on_host, n);
    SyclHelper::cpy_from_host(q, res_head, res_head_on_host, n);

    // call algorithm
    using kernel_name_1 = TestUtils::unique_kernel_name<class exclusive_scan_by_segment_1, (::std::size_t)alloc_type>;
    auto new_policy = oneapi::dpl::execution::make_device_policy<kernel_name_1>(q);
    oneapi::dpl::inclusive_scan_by_segment(new_policy, key_head, key_head+n, val_head, res_head,
        std::equal_to<uint64_t>(), std::plus<uint64_t>());

    // check values
    uint64_t check_value;
    for (int i = 0; i != 10; ++i) {
        if (i == 0 || key_head[i] != key_head[i-1])
            check_value = val_head[i];
        else
    	check_value += val_head[i];
        ASSERT_EQUAL(check_value, res_head[i]);
    }

    // call algorithm on single element range
    res_head[0] = 9;
    using kernel_name_2 = TestUtils::unique_kernel_name<class exclusive_scan_by_segment_2, (::std::size_t)alloc_type>;
    auto new_policy2 = oneapi::dpl::execution::make_device_policy<kernel_name_2>(q);
    oneapi::dpl::inclusive_scan_by_segment(new_policy2, key_head, key_head+1, val_head, res_head);

    SyclHelper::cpy_to_host(q, res_head_on_host, res_head, n);

    // check values
    ASSERT_EQUAL(1, res_head[0]);

    // Deallocate memory
    sycl::free(key_head, q);
    sycl::free(val_head, q);
    sycl::free(res_head, q);
}
#endif

void test_on_host()
{
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result, std::equal_to<int>(), std::plus<int>());

    // data is now {1, 2, 3, 1, 2, 1, 1, 2, 3, 4};
    ASSERT_EQUAL(result[0], 1);
    ASSERT_EQUAL(result[1], 2);
    ASSERT_EQUAL(result[2], 3);
    ASSERT_EQUAL(result[3], 1);
    ASSERT_EQUAL(result[4], 2);
    ASSERT_EQUAL(result[5], 1);
    ASSERT_EQUAL(result[6], 1);
    ASSERT_EQUAL(result[7], 2);
    ASSERT_EQUAL(result[8], 3);
    ASSERT_EQUAL(result[9], 4);
}

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
    test_with_usm<sycl::usm::alloc::shared>();
    test_with_usm<sycl::usm::alloc::device>();
#endif
    test_on_host();

    return TestUtils::done();
}
