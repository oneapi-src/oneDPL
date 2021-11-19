// -*- C++ -*-
//===-- extreme_value_distribution_test.cpp ---------------------------------===//
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
// Device copyable tests for distributions 

#include "support/utils.h"
#include <iostream>

#include "common_for_device_tests.h"

constexpr auto a = 40014u;
constexpr auto c = 200u;
constexpr auto m = 2147483563u;

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    sycl::queue queue(exception_handler);
    int err = 0;

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "bernoulli_distribution<sycl::vec<bool, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::bernoulli_distribution<sycl::vec<bool, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 1>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 3>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::exponential_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << "exponential_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::exponential_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 1>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 2>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 3>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 4>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 8>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    std::cout << "geometric_distribution<sycl::vec<int32_t, 16>> linear_congruential_engine" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::geometric_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::weibull_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "weibull_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::weibull_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_real_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_real_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "uniform_int_distribution<sycl::vec<int32_t, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::uniform_int_distribution<sycl::vec<int32_t, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 1>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
   err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 3>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    std::cout << "extreme_value_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::extreme_value_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 1>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 1>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 3>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 3>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "cauchy_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::cauchy_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::normal_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "normal_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::normal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>> oneapi::dpl::linear_congruential_engine
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "lognormal_distribution<sycl::vec<double, 2>> linear_congruential_engine" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 2>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>> oneapi::dpl::linear_congruential_engine
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "lognormal_distribution<sycl::vec<double, 4>> linear_congruential_engine" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 4>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>> oneapi::dpl::linear_congruential_engine
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "lognormal_distribution<sycl::vec<double, 8>> linear_congruential_engine" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 8>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // testing oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>> oneapi::dpl::linear_congruential_engine
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "lognormal_distribution<sycl::vec<double, 16>> linear_congruential_engine" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    err = test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<uint32_t,  a, c, m>>(queue);
#if TEST_LONG_RUN
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 16>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 8>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 4>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 3>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 2>,  a, c, m>>(queue);
    err += test<oneapi::dpl::lognormal_distribution<sycl::vec<double, 16>>, oneapi::dpl::linear_congruential_engine<sycl::vec<std::uint32_t, 1>,  a, c, m>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");


#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
