// -*- C++ -*-
//===-- sycl_policy.pass.cpp ----------------------------------------------===//
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
#include <vector>

#if _PSTL_BACKEND_SYCL
#include <CL/sycl.hpp>
#endif

#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if _PSTL_BACKEND_SYCL
template<typename Policy>
void test_policy_instance(const Policy& policy)
{
    auto __max_work_group_size = policy.queue().get_device().template get_info<cl::sycl::info::device::max_work_group_size>();
    EXPECT_TRUE(__max_work_group_size > 0, "policy: wrong work group size");
    auto __max_compute_units = policy.queue().get_device().template get_info<cl::sycl::info::device::max_compute_units>();
    EXPECT_TRUE(__max_compute_units > 0, "policy: wrong number of compute units");

    const int n = 10;
    const int value = -1;
    static ::std::vector<int> a(n);

    ::std::fill(a.begin(), a.end(), 0);
    ::std::fill(policy, a.begin(), a.end(), value);
#if _PSTL_SYCL_TEST_USM
    policy.queue().wait_and_throw();
#endif
    EXPECT_TRUE(::std::all_of(a.begin(), a.end(), [&value](int i) { return i == value; }), "wrong result of ::std::fill with policy");
}
#endif

int32_t
main()
{
#if _PSTL_BACKEND_SYCL
    PRINT_DEBUG("Test SYCL Policy(queue(default_selector))");
    test_policy_instance(
        oneapi::dpl::execution::make_sycl_policy<class Kernel_1>(cl::sycl::queue{TestUtils::default_selector}));
    PRINT_DEBUG("Test Policy(queue(default_selector))");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_2>(cl::sycl::queue{TestUtils::default_selector}));
    PRINT_DEBUG("Test Policy(default_selector)");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_3>(TestUtils::default_selector));
    PRINT_DEBUG("Test Policy(device(default_selector))");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_4>(cl::sycl::device{TestUtils::default_selector}));
    PRINT_DEBUG("Test Policy(ordered_queue(default_selector))");
    test_policy_instance(oneapi::dpl::execution::make_device_policy<class Kernel_5>(
        cl::sycl::queue{TestUtils::default_selector, cl::sycl::property::queue::in_order()}));
#endif

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}

