// -*- C++ -*-
//===-- scan_by_segment_cpu.pass.cpp ------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

#include <iostream>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

void test_inclusive_scan_by_segment_4() {
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result);

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

void test_inclusive_scan_by_segment_5() {
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result, std::equal_to<int>());

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

void test_inclusive_scan_by_segment_6() {
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

void test_exclusive_scan_by_segment_4() {
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result);

    // data is now {0, 1, 2, 0, 1, 0, 0, 1, 2, 3};

    ASSERT_EQUAL(result[0], 0);
    ASSERT_EQUAL(result[1], 1);
    ASSERT_EQUAL(result[2], 2);
    ASSERT_EQUAL(result[3], 0);
    ASSERT_EQUAL(result[4], 1);
    ASSERT_EQUAL(result[5], 0);
    ASSERT_EQUAL(result[6], 0);
    ASSERT_EQUAL(result[7], 1);
    ASSERT_EQUAL(result[8], 2);
    ASSERT_EQUAL(result[9], 3);
}

void test_exclusive_scan_by_segment_5() {
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int init = 5;

    oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result, init);

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

void test_exclusive_scan_by_segment_6() {
    int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
    int data[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int result[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int init = 5;

    oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::par, keys, keys + 10, data, result, init, std::equal_to<int>());

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

void test_exclusive_scan_by_segment_7() {
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
    test_inclusive_scan_by_segment_4();
    test_inclusive_scan_by_segment_5();
    test_inclusive_scan_by_segment_6();
    test_exclusive_scan_by_segment_4();
    test_exclusive_scan_by_segment_5();
    test_exclusive_scan_by_segment_6();
    test_exclusive_scan_by_segment_7();
    std::cout << "done" << std::endl;
    return 0;
}
