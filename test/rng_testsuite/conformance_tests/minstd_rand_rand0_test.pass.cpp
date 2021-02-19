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
// Test of minst_rand and minstd_rand0 engines - comparison of 10 000th element
//
// using minstd_rand0 = linear_congruential_engine<uint_fast32_t, 16’807, 0, 2’147’483’647>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type minstd_rand0
//     produces the value 1043618065.
//
// using minstd_rand = linear_congruential_engine<uint_fast32_t, 48’271, 0, 2’147’483’647>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type minstd_rand
//     produces the value 399268537

#include <iostream>
#include "support/utils.h"

#if _ONEDPL_BACKEND_SYCL
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // _ONEDPL_BACKEND_SYCL

int main() {

#if _ONEDPL_BACKEND_SYCL

    // Reference values
    uint_fast32_t minstd_rand0_ref_sample = 1043618065;
    uint_fast32_t minstd_rand_ref_sample = 399268537;

    // Generate 10 000th element for minstd_rand0
    auto minstd_rand0_sample       = test<oneapi::dpl::minstd_rand0,        10000, 1>();
    auto minstd_rand0_sample_vec2  = test<oneapi::dpl::minstd_rand0_vec<2>, 10000, 2>();
    // In case of minstd_rand0_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto minstd_rand0_sample_vec3  = test<oneapi::dpl::minstd_rand0_vec<3>, 10002, 3>();
    auto minstd_rand0_sample_vec4  = test<oneapi::dpl::minstd_rand0_vec<4>, 10000, 4>();
    auto minstd_rand0_sample_vec8  = test<oneapi::dpl::minstd_rand0_vec<8>, 10000, 8>();
    auto minstd_rand0_sample_vec16 = test<oneapi::dpl::minstd_rand0_vec<16>,10000, 16>();

    // Comparison
    std::cout << "\nThe 10000th reference value of minstd_rand0 engine is "            << minstd_rand0_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0 is "         << minstd_rand0_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0_vec<2> is "  << minstd_rand0_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0_vec<3> is "  << minstd_rand0_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0_vec<4> is "  << minstd_rand0_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0_vec<8> is "  << minstd_rand0_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand0_vec<16> is " << minstd_rand0_sample_vec16 << std::endl;
    if((minstd_rand0_ref_sample != minstd_rand0_sample)                   ||
        (minstd_rand0_ref_sample != minstd_rand0_sample_vec2)             ||
        (minstd_rand0_ref_sample != minstd_rand0_sample_vec3)             ||
        (minstd_rand0_ref_sample != minstd_rand0_sample_vec4)             ||
        (minstd_rand0_ref_sample != minstd_rand0_sample_vec8)             ||
        (minstd_rand0_ref_sample != minstd_rand0_sample_vec16)) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }


    auto minstd_rand_sample       = test<oneapi::dpl::minstd_rand,        10000, 1>();
    auto minstd_rand_sample_vec2  = test<oneapi::dpl::minstd_rand_vec<2>, 10000, 2>();
    // In case of minstd_rand_vec<3> engine generate 10002 values as 10000 % 3 != 0
    auto minstd_rand_sample_vec3  = test<oneapi::dpl::minstd_rand_vec<3>, 10002, 3>();
    auto minstd_rand_sample_vec4  = test<oneapi::dpl::minstd_rand_vec<4>, 10000, 4>();
    auto minstd_rand_sample_vec8  = test<oneapi::dpl::minstd_rand_vec<8>, 10000, 8>();
    auto minstd_rand_sample_vec16 = test<oneapi::dpl::minstd_rand_vec<16>,10000, 16>();

    // Comparison
    std::cout << "\nThe 10000th reference value of minstd_rand engine is "                    << minstd_rand_ref_sample  << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand is "                  << minstd_rand_sample << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand_vec<2> is "           << minstd_rand_sample_vec2 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand_vec<3> is "           << minstd_rand_sample_vec3 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand_vec<4> is "           << minstd_rand_sample_vec4 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand_vec<8> is "           << minstd_rand_sample_vec8 << std::endl;
    std::cout << "The 10000th produced value of oneapi::dpl::minstd_rand_vec<16> is "          << minstd_rand_sample_vec16 << std::endl;
    if((minstd_rand_ref_sample != minstd_rand_sample)                   ||
        (minstd_rand_ref_sample != minstd_rand_sample_vec2)             ||
        (minstd_rand_ref_sample != minstd_rand_sample_vec3)             ||
        (minstd_rand_ref_sample != minstd_rand_sample_vec4)             ||
        (minstd_rand_ref_sample != minstd_rand_sample_vec8)             ||
        (minstd_rand_ref_sample != minstd_rand_sample_vec16)) {
        std::cout << "Test FAILED" << std::endl;
        return 1;
    }

#else
    TestUtils::skip();

#endif // _ONEDPL_BACKEND_SYCL

    std::cout << "Test PASSED" << std::endl;
    return 0;
}
