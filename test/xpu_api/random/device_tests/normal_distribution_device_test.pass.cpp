// -*- C++ -*-
//===-- normal_distribution_device_test.pass.cpp ---------------------------------===//
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
//
// Abstract:
//
// Device copyable tests for normal distribution

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
#include "common_for_device_tests.h"
#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();
    int err = 0;

    // testing oneapi::dpl::normal_distribution<sycl::vec<float, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<float, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
//#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
//#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<float, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<float, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<float, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<float, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<float, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<float, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
    err += device_copyable_test<oneapi::dpl::normal_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  DefaultEngineParams::a, DefaultEngineParams::c, DefaultEngineParams::m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
