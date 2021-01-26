// -*- C++ -*-
//===-- reduce_by_segment_usm.pass.cpp --------------------------------------------===//
//
// Copyright (C) 2019 Intel Corporation
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
#include "oneapi/dpl/async"

#include <iostream>
#include <iomanip>

#include <CL/sycl.hpp>

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
        uint64_t* data1 = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* data2 = static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));

        //T data1[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
        //T data2[n1] = { 2, 3, 4, 5, 2, 2, 4, 4, 2, 2, 4, 4, 0 };

	// Initialize data
        for (int i = 0; i != n-1; ++i) {
            data1[i] = i % 4 + 1;
            data2[i] = i % 4 + 2;
            if (i > 3) {
                ++i;
                data1[i] = data1[i-1];
                data2[i] = data2[i-1];

            }
        }
        data1[n-1] = 0;
        data2[n-1] = 0;

        // call first algorithm
        auto new_policy = oneapi::dpl::execution::make_device_policy<class async1>(q);
        auto fut1 = oneapi::dpl::experimental::reduce_async(new_policy, data1, data1 + n);
        auto res1 = fut1.get();
        
        // check values
        ASSERT_EQUAL(res1, 26);
        
        // call second algorithm
        auto new_policy2 = oneapi::dpl::execution::make_device_policy<class async2>(q);
        auto res2 = oneapi::dpl::experimental::transform_reduce_async(new_policy2, data2, data2 + n, data1, 0, std::plus<>(), std::multiplies<>()).get();
        
        // check values
        ASSERT_EQUAL(res2, 96);
    }
    std::cout << "done" << std::endl;
    return 0;
}
