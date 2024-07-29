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

    // Alias for the oneapi::dpl::ext::ranges namespace
    namespace dpl_ranges = oneapi::dpl::ext::ranges;

    test_range_algo{}(dpl_ranges::find_if_not, std::ranges::find_if_not, pred);
    test_range_algo{}(dpl_ranges::find_if_not, std::ranges::find_if_not, pred, proj);
    test_range_algo<P2>{}(dpl_ranges::find_if_not, std::ranges::find_if_not, pred, &P2::x);
    test_range_algo<P2>{}(dpl_ranges::find_if_not, std::ranges::find_if_not, pred, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
