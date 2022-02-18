// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp --------------------------------------------===//
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
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include <iomanip>
#include <complex>

template<typename _T1, typename _T2>
void ASSERT_EQUAL(_T1&& X, _T2&& Y) {
    if(X!=Y)
        std::cout << "CHECK CORRECTNESS (REDUCE_BY_SEGMENT): fail (" << X << "," << Y << ")" << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>

template <typename KernelName, typename T>
void test_with_buffers()
{
    // create buffers
    sycl::buffer<T, 1> key_buf{ sycl::range<1>(13) };
    sycl::buffer<T, 1> val_buf{ sycl::range<1>(13) };
    sycl::buffer<T, 1> key_res_buf{ sycl::range<1>(13) };
    sycl::buffer<T, 1> val_res_buf{ sycl::range<1>(13) };

    {
        auto keys    = key_buf.template get_access<sycl::access::mode::read_write>();
        auto vals    = val_buf.template get_access<sycl::access::mode::read_write>();
        auto keys_res    = key_res_buf.template get_access<sycl::access::mode::read_write>();
        auto vals_res    = val_res_buf.template get_access<sycl::access::mode::read_write>();

        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
        //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

        // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, 0};
        // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

        // Initialize data
        for (int i = 0; i != 12; ++i) {
            keys[i] = i % 4 + 1;
            vals[i] = i % 4 + 1;
            keys_res[i] = 9;
            vals_res[i] = 1;
            if (i > 3) {
                ++i;
                keys[i] = keys[i-1];
                vals[i] = vals[i-1];
                keys_res[i] = 9;
                vals_res[i] = 1;
            }
        }
        keys[12] = 0;
        vals[12] = 0;
    }

    // create sycl iterators
    auto key_beg = oneapi::dpl::begin(key_buf);
    auto key_end = oneapi::dpl::end(key_buf);
    auto val_beg = oneapi::dpl::begin(val_buf);
    auto key_res_beg = oneapi::dpl::begin(key_res_buf);
    auto val_res_beg = oneapi::dpl::begin(val_res_buf);

    // create named policy from existing one
    auto new_policy = oneapi::dpl::execution::make_device_policy<KernelName>(
        oneapi::dpl::execution::dpcpp_default);

    // call algorithm
    auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_beg, key_end, val_beg, key_res_beg, val_res_beg);

    {
        // check values
        auto keys_res    = key_res_buf.template get_access<sycl::access::mode::read_write>();
        auto vals_res    = val_res_buf.template get_access<sycl::access::mode::read_write>();
        int n = std::distance(key_res_beg, res1.first);
        for (auto i = 0; i != n; ++i) {
            if (i < 4) {
                ASSERT_EQUAL(keys_res[i], T(i+1));
                ASSERT_EQUAL(vals_res[i], T(i+1));
            } else if (i == 4 || i == 6) {
                ASSERT_EQUAL(keys_res[i], T(1));
                ASSERT_EQUAL(vals_res[i], T(2));
            } else if (i == 5 || i == 7) {
                ASSERT_EQUAL(keys_res[i], T(3));
                ASSERT_EQUAL(vals_res[i], T(6));
            } else if (i == 8) {
                ASSERT_EQUAL(keys_res[i], T(0));
                ASSERT_EQUAL(vals_res[i], T(0));
            } else {
                std::cout << "fail: unexpected values in output range\n";
            }
        }

        // reset value_result for test using discard_iterator
        for (auto i = 0; i != n; ++i) {
            keys_res[i] = T(0);
            vals_res[i] = T(0);
        }
    }
}

template <typename KernelName, typename T>
void test_with_usm()
{
    sycl::queue q;
    int n = 13;

    // Allocate space for data using USM.
    T* key_head = static_cast<T*>(sycl::malloc_shared(n * sizeof(T), q.get_device(), q.get_context()));
    T* val_head = static_cast<T*>(sycl::malloc_shared(n * sizeof(T), q.get_device(), q.get_context()));
    T* key_res_head = static_cast<T*>(sycl::malloc_shared(n * sizeof(T), q.get_device(), q.get_context()));
    T* val_res_head = static_cast<T*>(sycl::malloc_shared(n * sizeof(T), q.get_device(), q.get_context()));

    //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };
    //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 };

    // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, 0};
    // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, 0};

    // Initialize data
    for (int i = 0; i != 12; ++i) {
        key_head[i] = i % 4 + 1;
        val_head[i] = i % 4 + 1;
        key_res_head[i] = T(9);
        val_res_head[i] = T(1);
        if (i > 3) {
            ++i;
            key_head[i] = key_head[i-1];
            val_head[i] = val_head[i-1];
            key_res_head[i] = T(9);
            val_res_head[i] = T(1);
        }
    }
    key_head[12] = T(0);
    val_head[12] = T(0);

    // call algorithm
    auto new_policy = oneapi::dpl::execution::make_device_policy<TestUtils::unique_kernel_name<KernelName, 1>>(q);
    auto res1 = oneapi::dpl::reduce_by_segment(new_policy, key_head, key_head + n, val_head, key_res_head, val_res_head);

    // check values
    n = std::distance(key_res_head, res1.first);
    for (auto i = 0; i != n; ++i) {
        if (i < 4) {
            ASSERT_EQUAL(key_res_head[i], T(i+1));
            ASSERT_EQUAL(val_res_head[i], T(i+1));
        } else if (i == 4 || i == 6) {
            ASSERT_EQUAL(key_res_head[i], T(1));
            ASSERT_EQUAL(val_res_head[i], T(2));
        } else if (i == 5 || i == 7) {
            ASSERT_EQUAL(key_res_head[i], T(3));
            ASSERT_EQUAL(val_res_head[i], T(6));
        } else if (i == 8) {
            ASSERT_EQUAL(key_res_head[i], T(0));
            ASSERT_EQUAL(val_res_head[i], T(0));
        } else {
            std::cout << "fail: unexpected values in output range\n";
        }
    }

    // call algorithm on single element range
    key_res_head[0] = T(9);
    val_res_head[0] = T(9);

    auto new_policy2 = oneapi::dpl::execution::make_device_policy<TestUtils::unique_kernel_name<KernelName, 2>>(q);
    auto res2 = oneapi::dpl::reduce_by_segment(new_policy2, key_head, key_head + 1, val_head, key_res_head, val_res_head);

    // check values
    n = std::distance(key_res_head, res2.first);
    ASSERT_EQUAL(n, 1);
    ASSERT_EQUAL(key_res_head[0], T(1));
    ASSERT_EQUAL(val_res_head[0], T(1));

    // Deallocate memory
    sycl::free(key_head, q);
    sycl::free(val_head, q);
    sycl::free(key_res_head, q);
    sycl::free(val_res_head, q);
}
#endif

template <typename T>
void test_on_host() {
    const int N = 7;
    T A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    T B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    T C[N];                         // output keys
    T D[N];                         // output values

    std::pair<T*, T*> new_end;
    new_end = oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, A, A + N, B, C, D, std::equal_to<T>(),
                                    std::plus<T>());
    // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
    // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.
    ASSERT_EQUAL(C[0], T(1));
    ASSERT_EQUAL(C[1], T(3));
    ASSERT_EQUAL(C[2], T(2));
    ASSERT_EQUAL(C[3], T(1));
    ASSERT_EQUAL(D[0], T(9));
    ASSERT_EQUAL(D[1], T(21));
    ASSERT_EQUAL(D[2], T(9));
    ASSERT_EQUAL(D[3], T(3));
    ASSERT_EQUAL(std::distance(C, new_end.first), 4);
    ASSERT_EQUAL(std::distance(D, new_end.second), 4);
}

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers<class KernelName1, std::uint64_t>();
    test_with_buffers<class KernelName2, std::complex<float>>();
    test_with_usm<class KernelName3, std::uint64_t>();
    test_with_usm<class KernelName4, std::complex<float>>();
#endif
    test_on_host<int>();
    test_on_host<std::complex<float>>();

    return TestUtils::done();
}
