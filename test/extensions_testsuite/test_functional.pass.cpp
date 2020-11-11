// -*- C++ -*-
//===-- test_functional.cpp -----------------------------------------------===//
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

#ifndef ONEDPL_STANDARD_POLICIES_ONLY
#include <CL/sycl.hpp>
#else
#include <vector>
#endif

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/iterator>

#include <oneapi/dpl/functional>

template<typename Iterator, typename T>
bool check_values(Iterator first, Iterator last, const T& val)
{
    return std::all_of(first, last,
                       [&val](const T& x) { return x == val; });
}


template<typename _T1, typename _T2> void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (PSTL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

int main() {

#ifdef ONEDPL_STANDARD_POLICIES_ONLY
    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 1; data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 1;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        auto t = std::copy_if( oneapi::dpl::execution::par, data.begin(), data.end(), result.begin(), oneapi::dpl::identity() );

        ASSERT_EQUAL(std::distance(result.begin(),t),4);
    }

    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = 3; data[1] = -1; data[2] = -4; data[3] = 1; data[4] = -5; data[5] = -9; data[6] = 2; data[7] = 6;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        std::exclusive_scan( oneapi::dpl::execution::par, data.begin(), data.begin() + 8, result.begin(), T(0) , oneapi::dpl::minimum<T>() );

        ASSERT_EQUAL(result[0], 0);
        ASSERT_EQUAL(result[1], 0);
        ASSERT_EQUAL(result[2], -1);
        ASSERT_EQUAL(result[3], -4);
        ASSERT_EQUAL(result[4], -4);
        ASSERT_EQUAL(result[5], -5);
        ASSERT_EQUAL(result[6], -9);
        ASSERT_EQUAL(result[7], -9);
    }

    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = -3; data[1] = 1; data[2] = 4; data[3] = -1; data[4] = 5; data[5] = 9; data[6] = -2; data[7] = 6;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        std::exclusive_scan( oneapi::dpl::execution::par, data.begin(), data.begin() + 8, result.begin(), T(0) , oneapi::dpl::maximum<T>() );

        ASSERT_EQUAL(result[0], 0);
        ASSERT_EQUAL(result[1], 0);
        ASSERT_EQUAL(result[2], 1);
        ASSERT_EQUAL(result[3], 4);
        ASSERT_EQUAL(result[4], 4);
        ASSERT_EQUAL(result[5], 5);
        ASSERT_EQUAL(result[6], 9);
        ASSERT_EQUAL(result[7], 9);

    }
#else
    /* IDENTITY TEST: */
    {
        using T = int;

        // create buffer
        cl::sycl::buffer<T, 1> src_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<T, 1> dst_buf{ cl::sycl::range<1>(8) };

        {
            auto data = src_buf.template get_access<cl::sycl::access::mode::write>();
            data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 1;
            data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 1;
        }

        auto data_it = oneapi::dpl::begin(src_buf);
        auto data_end_it = oneapi::dpl::end(src_buf);

        auto result_it = oneapi::dpl::begin(dst_buf);

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class IdentX>(oneapi::dpl::execution::dpcpp_default);

        auto t = std::copy_if( new_policy, data_it, data_end_it, result_it, oneapi::dpl::identity() );

        auto count = std::distance(result_it,t);

        ASSERT_EQUAL(count,4);
    }

    /* MAXIMUM TEST: */
    {
        using T = int;

        // create buffer
        cl::sycl::buffer<T, 1> src_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<T, 1> dst_buf{ cl::sycl::range<1>(8) };

        {   
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();
            auto dst = dst_buf.template get_access<cl::sycl::access::mode::write>();
            src[0] = -3; src[1] = 1; src[2] = 4; src[3] = -1; src[4] = 5; src[5] = 9; src[6] = -2; src[7] = 6;
            dst[0] = 0;  dst[1] = 0; dst[2] = 0;  dst[3] = 0; dst[4] = 0;  dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        
        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class MaxX>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::exclusive_scan(new_policy, src_it, src_end_it, dst_it, T(0), oneapi::dpl::maximum<T>());
        
        auto dst = dst_buf.template get_access<cl::sycl::access::mode::read>();
        ASSERT_EQUAL(dst[0], 0);
        ASSERT_EQUAL(dst[1], 0);
        ASSERT_EQUAL(dst[2], 1);
        ASSERT_EQUAL(dst[3], 4);
        ASSERT_EQUAL(dst[4], 4);
        ASSERT_EQUAL(dst[5], 5);
        ASSERT_EQUAL(dst[6], 9);
        ASSERT_EQUAL(dst[7], 9);
    }

    /* MINIMUM TEST: */
    {
        using T = int;

        // create buffer
        cl::sycl::buffer<T, 1> src_buf{ cl::sycl::range<1>(8) };
        cl::sycl::buffer<T, 1> dst_buf{ cl::sycl::range<1>(8) };
    
        {   
            auto src = src_buf.template get_access<cl::sycl::access::mode::write>();
            auto dst = dst_buf.template get_access<cl::sycl::access::mode::write>();

            src[0] = 3; src[1] = -1; src[2] = -4; src[3] = 1; src[4] = -5; src[5] = -9; src[6] = 2; src[7] = 6;
            dst[0] = 0;  dst[1] = 0; dst[2] = 0;  dst[3] = 0; dst[4] = 0;  dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        // create named policy from existing one
        auto new_policy = oneapi::dpl::execution::make_device_policy<class MinX>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::exclusive_scan(new_policy, src_it, src_end_it, dst_it, T(0), oneapi::dpl::minimum<T>());
    
        auto dst = dst_buf.template get_access<cl::sycl::access::mode::read>();
        ASSERT_EQUAL(dst[0], 0);
        ASSERT_EQUAL(dst[1], 0);
        ASSERT_EQUAL(dst[2], -1);
        ASSERT_EQUAL(dst[3], -4);
        ASSERT_EQUAL(dst[4], -4);
        ASSERT_EQUAL(dst[5], -5);
        ASSERT_EQUAL(dst[6], -9);
        ASSERT_EQUAL(dst[7], -9);
    }
#endif
    std::cout << "done\n";

    return 0;
}

