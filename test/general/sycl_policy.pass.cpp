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

#if _ONEDPL_BACKEND_SYCL
#include <CL/sycl.hpp>
#endif

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if _ONEDPL_BACKEND_SYCL
template<typename Policy>
void test_policy_instance(const Policy& policy)
{
    auto __max_work_group_size = policy.queue().get_device().template get_info<sycl::info::device::max_work_group_size>();
    EXPECT_TRUE(__max_work_group_size > 0, "policy: wrong work group size");
    auto __max_compute_units = policy.queue().get_device().template get_info<sycl::info::device::max_compute_units>();
    EXPECT_TRUE(__max_compute_units > 0, "policy: wrong number of compute units");

    const int n = 10;
    static ::std::vector<int> a(n);

    ::std::fill(a.begin(), a.end(), 0);
    ::std::fill(policy, a.begin(), a.end(), -1);
#if _PSTL_SYCL_TEST_USM
    policy.queue().wait_and_throw();
#endif
    EXPECT_TRUE(::std::all_of(a.begin(), a.end(), [](int i) { return i == -1; }), "wrong result of ::std::fill with policy");
}
#endif

int32_t
main()
{
#if _ONEDPL_BACKEND_SYCL
    PRINT_DEBUG("Test Policy(queue(default_selector))");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_2>(sycl::queue{TestUtils::default_selector}));
    PRINT_DEBUG("Test Policy(default_selector)");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_3>(TestUtils::default_selector));
    PRINT_DEBUG("Test Policy(device(default_selector))");
    test_policy_instance(
        oneapi::dpl::execution::make_device_policy<class Kernel_4>(sycl::device{TestUtils::default_selector}));
    PRINT_DEBUG("Test Policy(ordered_queue(default_selector))");
    test_policy_instance(oneapi::dpl::execution::make_device_policy<class Kernel_5>(
        sycl::queue{TestUtils::default_selector, sycl::property::queue::in_order()}));
#endif

    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}

