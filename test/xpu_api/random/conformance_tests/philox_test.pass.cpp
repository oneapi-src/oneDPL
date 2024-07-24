// -*- C++ -*-
//===-- minstd_rand_rand0_test.cpp -----------------------------------------===//
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
// Test for Philox random number generation engine - comparison of 10 000th element

#include "support/utils.h"
#include "support/test_config.h"
#include "common_for_conformance_tests.hpp"

#include <oneapi/dpl/random>

int main() {
    sycl::queue queue = TestUtils::get_test_queue();

    uint_fast32_t philox4_32_ref = 1955073260;
    uint_fast64_t philox4_64_ref = 3409172418970261260;
    int err = 0;

    // Generate 10 000th element for philox4_32
    err += test<oneapi::dpl::philox4x32, 10000, 1>(queue) != philox4_32_ref;
    std::cout << "\t\texpected: " << philox4_32_ref << std::endl;
    // Generate 10 000th element for philox4_64
    // err += test<oneapi::dpl::philox4x64, 10000, 1>(queue) != philox4_64_ref;
    // std::cout << "\t\texpected: " << philox4_64_ref << std::endl;

    EXPECT_TRUE(!err, "Test FAILED");

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}