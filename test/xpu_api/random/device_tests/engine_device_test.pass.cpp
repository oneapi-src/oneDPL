// -*- C++ -*-
//===-- engine_device_test.pass.cpp ---------------------------------===//
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
// Device copyable tests for engines 

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

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "linear_congruential_engine<48271, 0, 2147483647>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "linear_congruential_engine<16807, 0, 2147483647>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::minstd_rand0_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "subtract_with_carry_engine<24, 10, 24>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_base_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux24_base, 223, 23>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux24_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "subtract_with_carry_engine<48, 5, 12>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_base_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "discard_block_engine<ranlux48_base, 389, 11>" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48>(queue);
#if TEST_LONG_RUN
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<16>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<8>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<4>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<3>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<2>>(queue);
    err += device_copyable_test<oneapi::dpl::uniform_real_distribution<sycl::vec<float, 16>>, oneapi::dpl::ranlux48_vec<1>>(queue);
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_UNNAMED_LAMBDAS);
}
