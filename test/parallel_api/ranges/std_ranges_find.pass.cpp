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

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto find_checker = TEST_PREPARE_CALLABLE(std::ranges::find);

    test_range_algo<0>{big_sizes_per_policy}(dpl_ranges::find, find_checker, 4);  //found case
    test_range_algo<1>{}(dpl_ranges::find, find_checker, -1); //not found case
    test_range_algo<2>{}(dpl_ranges::find, find_checker, 4, proj);
    test_range_algo<3, P2>{}(dpl_ranges::find, find_checker, 4, &P2::x);
    test_range_algo<4, P2>{}(dpl_ranges::find, find_checker, 4, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
