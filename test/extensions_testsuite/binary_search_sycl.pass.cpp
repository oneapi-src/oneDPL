// -*- C++ -*-
//===-- lower_bound_sycl.pass.cpp --------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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

#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

int main()
{
    bool correctness_flag =true;

    //Test case 1
    cl::sycl::buffer<uint64_t, 1> _key_buf{ cl::sycl::range<1>(10) };
    cl::sycl::buffer<uint64_t, 1> _val_buf{ cl::sycl::range<1>(5) };
    cl::sycl::buffer<uint64_t, 1> _res_buf{ cl::sycl::range<1>(5) };
    { 
        auto key_buf = _key_buf.template get_access<cl::sycl::access::mode::read_write>();
	auto val_buf = _val_buf.template get_access<cl::sycl::access::mode::read_write>();
	auto res_buf = _res_buf.template get_access<cl::sycl::access::mode::read_write>();
     
	// Initialize data
	key_buf[0] = 0; key_buf[1] = 2; key_buf[2] = 2; key_buf[3] = 2; key_buf[4] = 3;
	key_buf[5] = 3; key_buf[6] = 3; key_buf[7] = 3; key_buf[8] = 6; key_buf[9] = 6;
	
	val_buf[0] = 0; val_buf[1] = 2; val_buf[2] = 4; val_buf[3] = 7; val_buf[4] = 6;
    }
    
    // create sycl iterators
    auto key_beg = oneapi::dpl::begin(_key_buf);
    auto key_end = oneapi::dpl::end(_key_buf);
    auto val_beg = oneapi::dpl::begin(_val_buf);
    auto val_end = oneapi::dpl::end(_val_buf);
    auto res_beg = oneapi::dpl::begin(_res_buf);

    // create named policy from existing one
    auto new_policy = oneapi::dpl::execution::make_device_policy<class binarySearch>(oneapi::dpl::execution::dpcpp_default);
    
    // call algorithm
    oneapi::dpl::binary_search(new_policy, key_beg, key_end, val_beg , val_end, res_beg);
    
    auto res = _res_buf.template get_access<cl::sycl::access::mode::read>();
    
    //check data
    if((res[0] != true) || (res[1] != true) || (res[2] != false) && (res[3] != false) && (res[4] != true))
        correctness_flag = false;

    //test case 2
    cl::sycl::buffer<uint64_t, 1> _key_buf_2{ cl::sycl::range<1>(2) };
    cl::sycl::buffer<uint64_t, 1> _res_buf_2{ cl::sycl::range<1>(5) };
    {
        auto key_buf_2 = _key_buf_2.template get_access<cl::sycl::access::mode::read_write>();
	
	// Initialize data
	key_buf_2[0] = 0; key_buf_2[1] = 2;
    }

    // create sycl iterators
    auto key_beg_2 = oneapi::dpl::begin(_key_buf_2);
    auto key_end_2 = oneapi::dpl::end(_key_buf_2);
    auto res_beg_2 = oneapi::dpl::begin(_res_buf_2);
    
    // create named policy from existing one
    auto new_policy2 = oneapi::dpl::execution::make_device_policy<class binarySearch2>(oneapi::dpl::execution::dpcpp_default);

    // call algorithm
    oneapi::dpl::binary_search(new_policy2, key_beg_2, key_end_2, val_beg , val_end, res_beg_2, std::less<int>());

    auto res_2 = _res_buf_2.template get_access<cl::sycl::access::mode::read>();
    //check data
    if((res_2[0] != true) || (res_2[1]!= true) || (res_2[2] != false) || (res_2[3] != false) || (res_2[4] != false))
        correctness_flag = false;
    
    if(correctness_flag == true)
        std::cout << "done" << std::endl;
    else
        std::cout << "Values do not match." << std::endl;
    
    return 0;
}
