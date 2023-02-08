// -*- C++ -*-
//===-- cauchy_distribution_device_test.pass.cpp ---------------------------------===//
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
// Device copyable tests for cauchy distribution

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS
#include "common_for_device_tests.h"
#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();
    int err = 0;

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<float, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<std::uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += device_copyable_test<oneapi::dpl::cauchy_distribution<sycl::vec<float, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
