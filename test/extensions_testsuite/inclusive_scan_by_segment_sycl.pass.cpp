// -*- C++ -*-
//===-- inclusive_scan_by_segment_sycl.pass.cpp ------------------------------------===//
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

class InclusiveScanBySegment {};     // name for policy

int main() {

    // #4 INCLUSIVE SCAN BY SEGMENT TEST //

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
        auto new_policy = oneapi::dpl::execution::make_device_policy<InclusiveScanBySegment>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm
        oneapi::dpl::inclusive_scan_by_segment(new_policy, key_beg, key_end, val_beg, res_beg,
            std::equal_to<uint64_t>(), std::plus<uint64_t>());

        // check values
        auto key_acc = _key_buf.get_access<cl::sycl::access::mode::read>();
        auto val_acc = _val_buf.get_access<cl::sycl::access::mode::read>();
        auto res_acc = _res_buf.get_access<cl::sycl::access::mode::read>();
        uint64_t check_value;
        for (int i = 0; i != 10; ++i) {
            if (i == 0 || key_acc[i] != key_acc[i-1])
                check_value = val_acc[i];
            else
                check_value += val_acc[i];
            ASSERT_EQUAL(check_value, res_acc[i]);
        }
    }
    std::cout << "done" << std::endl;
    return 0;
}
