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
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    constexpr int rotate_val = 6;

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(max_n);

    auto src = views::iota(0, max_n);
    auto res = rotate_copy(TestUtils::default_dpcpp_policy, src, rotate_val, A);

    //check result
    EXPECT_TRUE(res == max_n, "wrong result from rotate_copy");
    EXPECT_EQ_RANGES(src | views::rotate(rotate_val), views::host_all(A), "wrong effect from rotate_copy");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
