// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

template<typename _Policy>
void
test_with_std_pol(_Policy&& policy)
{
    const int n = 1000000;
    std::vector<int> keys_buf(n); //keys
    std::vector<int> vals_buf(n); //values

    // create objects to iterate over buffers
    auto keys_begin = keys_buf.begin();
    auto vals_begin = vals_buf.begin();

    auto counting_begin = oneapi::dpl::counting_iterator<int>{0};

    // 1. Initialization of buffers
    std::transform(policy, counting_begin, counting_begin + n, keys_begin,
                   [n](int i) { return i%2 + 1; });
    // fill vals_buf with the analogue of std::iota using counting_iterator
    std::copy(policy, counting_begin, counting_begin + n, vals_begin);

    // 2. Sorting
    // stable sort by keys
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, std::less<void>());

    // 3.Checking results
    auto k = (n-1)/2 + 1;
    for (int i = 0; i < n; ++i)
    {
        if(i-k < 0)
        {
             //a key should be 1 and value should be even
            EXPECT_TRUE(keys_buf[i] == 1 && vals_buf[i] % 2 == 0, "wrong sort_by_key result with a standard policy");
        }
        else
        {
            //a key should be 2 and value should be odd
            EXPECT_TRUE(keys_buf[i] == 2 && vals_buf[i] % 2 == 1, "wrong sort_by_key result with a standard policy");
        }
    }
}

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type>
void
test_with_usm(sycl::queue& q)
{
    constexpr int N = 32;

    int h_key[N] = {};
    int h_val[N] = {};
    for (int i = 0; i < N; i++)
    {
        h_val[i] = ((N - 1 - i) / 3) * 3;
        h_key[i] = i * 10;
    }

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_key(q, ::std::begin(h_key), ::std::end(h_key));
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(q, ::std::begin(h_val), ::std::end(h_val));

    int* d_key = dt_helper_h_key.get_data();
    int* d_val = dt_helper_h_val.get_data();

    auto myPolicy = TestUtils::make_device_policy<
        TestUtils::unique_kernel_name<class copy, TestUtils::uniq_kernel_index<alloc_type>()>>(q);
    oneapi::dpl::sort_by_key(myPolicy, d_key, d_key + N, d_val, std::greater<void>());

    int h_skey[N] = {};
    int h_sval[N] = {};

    dt_helper_h_key.retrieve_data(h_skey);
    dt_helper_h_val.retrieve_data(h_sval);

    for (int i = 0; i < N; i++)
    {
        if (i < (N - 1))
        {
            EXPECT_TRUE(h_skey[i] >= h_skey[i + 1], "wrong sort result with hetero policy, USM data");
        }
    }
}

void
test_with_buffers(sycl::queue& q)
{
    const int n = 1000000;
    sycl::buffer<int> keys_buf{n};  // buffer with keys
    sycl::buffer<int> vals_buf{n};  // buffer with values

    auto policy = TestUtils::make_device_policy(q);

    // create objects to iterate over buffers
    auto keys_begin = oneapi::dpl::begin(keys_buf);
    auto vals_begin = oneapi::dpl::begin(vals_buf);

    auto counting_begin = oneapi::dpl::counting_iterator<int>{0};

    // 1. Initialization of buffers
    std::transform(policy, counting_begin, counting_begin + n, keys_begin,
                   [n](int i) { return i%2 + 1; });
    // fill vals_buf with the analogue of std::iota using counting_iterator
    std::copy(policy, counting_begin, counting_begin + n, vals_begin);

    // 2. Sorting
    // stable sort by keys
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, std::less<void>());

    // 3.Checking results
    sycl::host_accessor host_keys(keys_buf, sycl::read_only);
    sycl::host_accessor host_vals(vals_buf, sycl::read_only);

    // expected output:
    auto k = (n-1)/2 + 1;
    for (int i = 0; i < n; ++i)
    {
        if(i-k < 0)
        {
             //a key should be 1 and value should be even
            EXPECT_TRUE(host_keys[i] == 1 && host_vals[i] % 2 == 0,
                "wrong sort_by_key result with a herero policy, sycl buffer");
        }
        else
        {
            //a key should be 2 and value should be odd
            EXPECT_TRUE(host_keys[i] == 2 && host_vals[i] % 2 == 1,
                "wrong sort_by_key result with a herero policy, sycl buffer");
        }
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q = TestUtils::get_test_queue();
#if _ONEDPL_DEBUG_SYCL
    std::cout << "    Device Name = " << q.get_device().get_info<sycl::info::device::name>().c_str() << "\n";
#endif // _ONEDPL_DEBUG_SYCL

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared>(q);
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device>(q);
    // Run tests for sycl buffers
    test_with_buffers(q);
#endif // TEST_DPCPP_BACKEND_PRESENT

#if !TEST_DPCPP_BACKEND_PRESENT
    test_with_std_pol(oneapi::dpl::execution::seq);
    test_with_std_pol(oneapi::dpl::execution::unseq);
    test_with_std_pol(oneapi::dpl::execution::par);
    test_with_std_pol(oneapi::dpl::execution::par_unseq);
#endif // !TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
