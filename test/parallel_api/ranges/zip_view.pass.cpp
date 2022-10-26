// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <oneapi/dpl/execution>

#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    char data[max_n] = {'b', 'e', 'g', 'f', 'c', 'd', 'a', 'j', 'i', 'h'};
    int key[max_n] = {1, 4, 6, 5, 2, 3, 0, 9, 8, 7};

    using namespace oneapi::dpl::experimental::ranges;

    auto view = nano::views::all(data);
    auto z = zip_view(nano::views::all(data), nano::views::all(key));

    //check access
    EXPECT_TRUE(::std::get<0>(z[2]) == 'g', "wrong effect with zip_view");

    const size_t max_int32p2 = (size_t)::std::numeric_limits<int32_t>::max() + 2UL;
    ::std::vector<char> large_data(max_int32p2);
    ::std::vector<char> large_keys(max_int32p2);

    auto large_z = zip_view(nano::views::all(large_data), nano::views::all(large_keys));
    sycl::queue q{};

    //check that zip_view ranges can be larger than a signed 32 bit integer
    size_t i = large_data.size() - 1;

    large_data[i] = i % ::std::numeric_limits<char>::max();
    large_keys[i] = (i + 1) % ::std::numeric_limits<char>::max();

    char expected_key = large_keys[i];
    char actual_key = ::std::get<1>(large_z[i]);
    EXPECT_EQ(expected_key, actual_key, "wrong effect with zip_view bracket operator");
    char expected_data = large_data[i];
    char actual_data = ::std::get<0>(large_z[i]);
    EXPECT_EQ(expected_data, actual_data, "wrong effect with zip_view bracket operator");

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
