// -*- C++ -*-
//===-- reduce_by_segment_cpu.pass.cpp ------------------------------------===//
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
        std::cout << "CHECK CORRECTNESS (DPSTD WITH CPU): fail (" << X << "," << Y << ")" << std::endl;
}

void test_reduce_by_segment_5() {
    const int N = 7;
    int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    int C[N];                         // output keys
    int D[N];                         // output values
  
    std::pair<int*,int*> new_end;
    new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, A, A + N, B, C, D);

    // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
    // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
    ASSERT_EQUAL(C[0], 1);
    ASSERT_EQUAL(C[1], 3);
    ASSERT_EQUAL(C[2], 2);
    ASSERT_EQUAL(C[3], 1);
    ASSERT_EQUAL(D[0], 9);
    ASSERT_EQUAL(D[1], 21);
    ASSERT_EQUAL(D[2], 9);
    ASSERT_EQUAL(D[3], 3);
    ASSERT_EQUAL(std::distance(C, new_end.first), 4);
    ASSERT_EQUAL(std::distance(D, new_end.second), 4);
}

void test_reduce_by_segment_6() {
    const int N = 7;
    int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    int C[N];                         // output keys
    int D[N];                         // output values

    std::pair<int*,int*> new_end;
    new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, A, A + N, B, C, D, std::equal_to<int>());
    // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
    // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
    ASSERT_EQUAL(C[0], 1);
    ASSERT_EQUAL(C[1], 3);
    ASSERT_EQUAL(C[2], 2);
    ASSERT_EQUAL(C[3], 1);
    ASSERT_EQUAL(D[0], 9);
    ASSERT_EQUAL(D[1], 21);
    ASSERT_EQUAL(D[2], 9);
    ASSERT_EQUAL(D[3], 3);
    ASSERT_EQUAL(std::distance(C, new_end.first), 4);
    ASSERT_EQUAL(std::distance(D, new_end.second), 4);
}

void test_reduce_by_segment_7() {
    const int N = 7;
    int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    int C[N];                         // output keys
    int D[N];                         // output values

    std::pair<int*,int*> new_end;
    new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, A, A + N, B, C, D, std::equal_to<int>(),
                                    std::plus<int>());
    // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
    // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.

}

int main() {
    test_reduce_by_segment_5();
    test_reduce_by_segment_6();
    test_reduce_by_segment_7();
    std::cout << "done" << std::endl;
    return 0;
}

