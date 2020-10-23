// -*- C++ -*-
//===-- exclusive_scan_by_segment_usm.pass.cpp ------------------------------------===//
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

int main() {

    cl::sycl::queue q;
    const int n = 10;

    // #1 EXCLUSIVE SCAN BY SEGMENT TEST //

    {
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
        auto new_policy = oneapi::dpl::execution::make_device_policy(q);
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
        auto new_policy2 = oneapi::dpl::execution::make_device_policy(q);
        oneapi::dpl::exclusive_scan_by_segment(new_policy2, key_head, key_head+1, val_head, res_head,
            (uint64_t)0);

	// check values
        ASSERT_EQUAL(0, res_head[0]);
    }
    std::cout << "done" << std::endl;
    return 0;
}
