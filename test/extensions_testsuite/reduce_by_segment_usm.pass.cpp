// -*- C++ -*-
//===-- reduce_by_segment_usm.pass.cpp --------------------------------------------===//
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
#include <iomanip>

#include <CL/sycl.hpp>

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

// comparator implementing operator==
template<typename T>
class my_equal
{
public:
    using first_argument_type = T;
    using second_argument_type = T;

    explicit my_equal() {}

    template <typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const {
        return std::forward<_Xp>(__x) == std::forward<_Yp>(__y);
    }
};

// binary functor implementing operator+
class my_plus
{
public:
    explicit my_plus() {}

    template <typename _Xp, typename _Yp>
    bool operator()(_Xp&& __x, _Yp&& __y) const {
        return std::forward<_Xp>(__x) + std::forward<_Yp>(__y);
    }
};

int main() {

    cl::sycl::queue q;
    const int n = 13;

    // #6 REDUCE BY SEGMENT TEST //

    {
        // Allocate space for data using USM.
        uint64_t* key_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* val_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* key_res_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* val_res_head = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));

        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
        //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

        // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, 0};
        // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

	// Initialize data
        for (int i = 0; i != 12; ++i) {
            key_head[i] = i % 4 + 1;
            val_head[i] = i % 4 + 1;
            key_res_head[i] = 9;
            val_res_head[i] = 1;
            if (i > 3) {
                ++i;
                key_head[i] = key_head[i-1];
                val_head[i] = val_head[i-1];
                key_res_head[i] = 9;
                val_res_head[i] = 1;
            }
        }
        key_head[12] = 0;
        val_head[12] = 0;

        // call algorithm
        auto new_policy = oneapi::dpl::execution::make_device_policy(q);
        auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_head, key_head + n, val_head, key_res_head, val_res_head);

	// check values
        int n = std::distance(key_res_head, res1.first);
        for (auto i = 0; i != n; ++i) {
            if (i < 4) {
                ASSERT_EQUAL(key_res_head[i], i+1);
                ASSERT_EQUAL(val_res_head[i], i+1);
            } else if (i == 4 || i == 6) {
                ASSERT_EQUAL(key_res_head[i], 1);
                ASSERT_EQUAL(val_res_head[i], 2);
            } else if (i == 5 || i == 7) {
                ASSERT_EQUAL(key_res_head[i], 3);
                ASSERT_EQUAL(val_res_head[i], 6);
            } else if (i == 8) {
                ASSERT_EQUAL(key_res_head[i], 0);
                ASSERT_EQUAL(val_res_head[i], 0);
            } else {
                std::cout << "fail: unexpected values in output range\n";
            }
        }

        // call algorithm on single element range
        key_res_head[0] = 9;
        val_res_head[0] = 9;

        auto new_policy2 = oneapi::dpl::execution::make_device_policy(q);
        auto res2 = oneapi::dpl::reduce_by_segment(new_policy2, key_head, key_head + 1, val_head, key_res_head, val_res_head);

	// check values
        n = std::distance(key_res_head, res2.first);
        ASSERT_EQUAL(n, 1);
        ASSERT_EQUAL(key_res_head[0], 1);
        ASSERT_EQUAL(val_res_head[0], 1);
    }
    std::cout << "done" << std::endl;
    return 0;
}
