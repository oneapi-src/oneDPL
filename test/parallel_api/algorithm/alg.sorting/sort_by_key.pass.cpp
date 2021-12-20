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

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type>
void
test_with_usm(sycl::queue q)
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

    auto first = oneapi::dpl::make_zip_iterator(d_key, d_val);
    auto last = first + N;

    auto myPolicy = oneapi::dpl::execution::make_device_policy<
        TestUtils::unique_kernel_name<class copy, TestUtils::uniq_kernel_index<alloc_type>()>>(q);
    std::sort(myPolicy, first, last,
              [](const auto& item1, const auto& item2) { return std::get<0>(item1) > std::get<0>(item2); });

    int h_skey[N] = {};
    int h_sval[N] = {};

    dt_helper_h_key.retrieve_data(h_skey);
    dt_helper_h_val.retrieve_data(h_sval);

    for (int i = 0; i < N; i++)
    {
        if (i < (N - 1))
        {
            EXPECT_TRUE(h_skey[i] >= h_skey[i + 1], "wrong sort result");
        }
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q;
#if _ONEDPL_DEBUG_SYCL
    std::cout << "    Device Name = " << q.get_device().get_info<cl::sycl::info::device::name>().c_str() << "\n";
#endif // _ONEDPL_DEBUG_SYCL

    // Run tests for USM shared memory
    test_with_usm<sycl::usm::alloc::shared>(q);
    // Run tests for USM device memory
    test_with_usm<sycl::usm::alloc::device>(q);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
