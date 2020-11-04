// -*- C++ -*-
//===-- ranlux_24_48_test.cpp ----------------------------------------------===//
//
// Copyright (C) 2020 Intel Corporation
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
// Test of ranlux24 and ranlux48 engines - comparison of 10 000th element
//
// using ranlux24 = discard_block_engine<ranlux24_base, 223, 23>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux24
//     produces the value 9901578.
//
// using ranlux48 = discard_block_engine<ranlux48_base, 389, 11>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux48
//     produces the value 249142670248501

#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>

int main() {

    // Reference values
    uint_fast32_t ranlux24_ref_sample = 9901578;
    uint_fast64_t ranlux48_ref_sample = 249142670248501;

    // Generate 10 000th element for ranlux24
    auto ranlux24_sample       = test<oneapi::dpl::ranlux24,        10000, 1>();
    auto ranlux24_sample_vec2  = test<oneapi::dpl::ranlux24_vec<2>, 10000, 2>();
    // In case of ranlux24_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto ranlux24_sample_vec3  = test<oneapi::dpl::ranlux24_vec<3>, 10002, 3>();
    auto ranlux24_sample_vec4  = test<oneapi::dpl::ranlux24_vec<4>, 10000, 4>();
    auto ranlux24_sample_vec8  = test<oneapi::dpl::ranlux24_vec<8>, 10000, 8>();
    auto ranlux24_sample_vec16 = test<oneapi::dpl::ranlux24_vec<16>,10000, 16>();

    // Comparison
    std::cout << "\nThe 10000th reference value of ranlux24 engine is "            << ranlux24_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24 is "         << ranlux24_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_vec<2> is "  << ranlux24_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_vec<3> is "  << ranlux24_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_vec<4> is "  << ranlux24_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_vec<8> is "  << ranlux24_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux24_vec<16> is " << ranlux24_sample_vec16 << std::endl;
    if((ranlux24_ref_sample == ranlux24_sample)                   &&
        (ranlux24_ref_sample == ranlux24_sample_vec2)             &&
        (ranlux24_ref_sample == ranlux24_sample_vec3)             &&
        (ranlux24_ref_sample == ranlux24_sample_vec4)             &&
        (ranlux24_ref_sample == ranlux24_sample_vec8)             &&
        (ranlux24_ref_sample == ranlux24_sample_vec16)) {
        std::cout << "Test PASSED" << std::endl;
    }
    else {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    auto ranlux48_sample       = test<oneapi::dpl::ranlux48,        10000, 1>();
    auto ranlux48_sample_vec2  = test<oneapi::dpl::ranlux48_vec<2>, 10000, 2>();
    // In case of ranlux48_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto ranlux48_sample_vec3  = test<oneapi::dpl::ranlux48_vec<3>, 10002, 3>();
    auto ranlux48_sample_vec4  = test<oneapi::dpl::ranlux48_vec<4>, 10000, 4>();
    auto ranlux48_sample_vec8  = test<oneapi::dpl::ranlux48_vec<8>, 10000, 8>();
    auto ranlux48_sample_vec16 = test<oneapi::dpl::ranlux48_vec<16>,10000, 16>();

    std::cout << "\nThe 10000th reference value of ranlux48 engine is "                    << ranlux48_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48 is "                  << ranlux48_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_vec<2> is "           << ranlux48_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_vec<3> is "           << ranlux48_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_vec<4> is "           << ranlux48_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_vec<8> is "           << ranlux48_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::ranlux48_vec<16> is "          << ranlux48_sample_vec16 << std::endl;
    if((ranlux48_ref_sample == ranlux48_sample)                   &&
        (ranlux48_ref_sample == ranlux48_sample_vec2)             &&
        (ranlux48_ref_sample == ranlux48_sample_vec3)             &&
        (ranlux48_ref_sample == ranlux48_sample_vec4)             &&
        (ranlux48_ref_sample == ranlux48_sample_vec8)             &&
        (ranlux48_ref_sample == ranlux48_sample_vec16)) {
        std::cout << "Test PASSED" << std::endl;
    }
    else {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

    return 0;
}