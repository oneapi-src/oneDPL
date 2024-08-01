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

    test_range_algo<int, data_in_in>{}(dpl_ranges::equal, std::ranges::equal, binary_pred);
    test_range_algo<int, data_in_in>{}(dpl_ranges::equal, std::ranges::equal, binary_pred, proj, proj);
    test_range_algo<P2, data_in_in>{}(dpl_ranges::equal, std::ranges::equal, binary_pred, &P2::x, &P2::x);
    test_range_algo<P2, data_in_in>{}(dpl_ranges::equal, std::ranges::equal, binary_pred, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
