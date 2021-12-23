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

#include <numeric>

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type>
void
test_with_usm(sycl::queue& q)
{
    constexpr ::std::size_t N = 16;

    auto prepare_data = [](::std::size_t* idx, int* val)
    {
        for (int i = 0; i < N; i++)
        {
            idx[i] = i + 1;
            val[i] = 0;
        }
    };

    // Prepare source data
    ::std::size_t h_idx[N] = {};
    int           h_val[N] = {};
    prepare_data(h_idx, h_val);

    // Copy source data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, ::std::size_t> dt_helper_h_idx(q, ::std::begin(h_idx), ::std::end(h_idx));
    auto d_idx = dt_helper_h_idx.get_data();

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(q, ::std::begin(h_val), ::std::end(h_val));
    auto d_val = dt_helper_h_val.get_data();

    // Run dpl::exclusive_scan algorithm on USM shared-device memory
    auto myPolicy = oneapi::dpl::execution::make_device_policy<
        TestUtils::unique_kernel_name<class copy, (::std::size_t)alloc_type>>(q);
    oneapi::dpl::exclusive_scan(myPolicy, d_idx, d_idx + N, d_val, 0);

    // Copy results from USM shared/device memory to host
    ::std::size_t h_sidx[N] = {};
    int           h_sval[N] = {};
    dt_helper_h_idx.retrieve_data(h_sidx);
    dt_helper_h_val.retrieve_data(h_sval);

    // Check results
    ::std::size_t h_sidx_expected[N] = {};
    int           h_sval_expected[N] = {};
    prepare_data(h_sidx_expected, h_sval_expected);
    ::std::exclusive_scan(h_sidx_expected, h_sidx_expected + N, h_sval_expected, 0);

    EXPECT_EQ_N(h_sidx_expected, h_sidx, N, "wrong effect from exclusive_scan - h_sidx");
    EXPECT_EQ_N(h_sval_expected, h_sval, N, "wrong effect from exclusive_scan - h_sval");
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
