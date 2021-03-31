// -*- C++ -*-
//===-- ranlux_24_48_base_test.cpp ----------------------------------------===//
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
// Test of ranlux24_base and ranlux48_base engines - comparison of 10 000th element
//
// using ranlux24_base = subtract_with_carry_engine<uint_fast32_t, 24, 10, 24>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux24_base
//     produces the value 7937952.
//
// using ranlux48_base = subtract_with_carry_engine<uint_fast64_t, 48, 5, 12>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux48_base
//     produces the value 61839128582725

#include "support/utils.h"
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {

#if TEST_DPCPP_BACKEND_PRESENT

    // Reference values
    uint_fast32_t ranlux24_base_ref_sample = 7937952;
    uint_fast64_t ranlux48_base_ref_sample = 61839128582725;

    // Generate 10 000th element for ranlux24_base
    auto ranlux24_base_sample       = test<oneapi::dpl::ranlux24_base,        10000, 1>();
    auto ranlux24_base_sample_vec2  = test<oneapi::dpl::ranlux24_base_vec<2>, 10000, 2>();
    // In case of ranlux24_base_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto ranlux24_base_sample_vec3  = test<oneapi::dpl::ranlux24_base_vec<3>, 10002, 3>();
    auto ranlux24_base_sample_vec4  = test<oneapi::dpl::ranlux24_base_vec<4>, 10000, 4>();
    auto ranlux24_base_sample_vec8  = test<oneapi::dpl::ranlux24_base_vec<8>, 10000, 8>();
    auto ranlux24_base_sample_vec16 = test<oneapi::dpl::ranlux24_base_vec<16>,10000, 16>();

    // Comparison
    std::cout << "\nThe 10000th reference value of ranlux24_base engine is "            << ranlux24_base_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base is "         << ranlux24_base_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base_vec<2> is "  << ranlux24_base_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base_vec<3> is "  << ranlux24_base_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base_vec<4> is "  << ranlux24_base_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base_vec<8> is "  << ranlux24_base_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_base_vec<16> is " << ranlux24_base_sample_vec16 << std::endl;
    if((ranlux24_base_ref_sample != ranlux24_base_sample)                   ||
        (ranlux24_base_ref_sample != ranlux24_base_sample_vec2)             ||
        (ranlux24_base_ref_sample != ranlux24_base_sample_vec3)             ||
        (ranlux24_base_ref_sample != ranlux24_base_sample_vec4)             ||
        (ranlux24_base_ref_sample != ranlux24_base_sample_vec8)             ||
        (ranlux24_base_ref_sample != ranlux24_base_sample_vec16)) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    auto ranlux48_base_sample       = test<oneapi::dpl::ranlux48_base,        10000, 1>();
    auto ranlux48_base_sample_vec2  = test<oneapi::dpl::ranlux48_base_vec<2>, 10000, 2>();
    // In case of ranlux48_base_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto ranlux48_base_sample_vec3  = test<oneapi::dpl::ranlux48_base_vec<3>, 10002, 3>();
    auto ranlux48_base_sample_vec4  = test<oneapi::dpl::ranlux48_base_vec<4>, 10000, 4>();
    auto ranlux48_base_sample_vec8  = test<oneapi::dpl::ranlux48_base_vec<8>, 10000, 8>();
    auto ranlux48_base_sample_vec16 = test<oneapi::dpl::ranlux48_base_vec<16>,10000, 16>();

    // Comparison
    std::cout << "\nThe 10000th reference value of ranlux48_base engine is "                    << ranlux48_base_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base is "                  << ranlux48_base_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base_vec<2> is "           << ranlux48_base_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base_vec<3> is "           << ranlux48_base_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base_vec<4> is "           << ranlux48_base_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base_vec<8> is "           << ranlux48_base_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_base_vec<16> is "          << ranlux48_base_sample_vec16 << std::endl;
    if((ranlux48_base_ref_sample != ranlux48_base_sample)                   ||
        (ranlux48_base_ref_sample != ranlux48_base_sample_vec2)             ||
        (ranlux48_base_ref_sample != ranlux48_base_sample_vec3)             ||
        (ranlux48_base_ref_sample != ranlux48_base_sample_vec4)             ||
        (ranlux48_base_ref_sample != ranlux48_base_sample_vec8)             ||
        (ranlux48_base_ref_sample != ranlux48_base_sample_vec16)) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
