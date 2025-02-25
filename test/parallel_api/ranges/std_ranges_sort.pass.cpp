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

    auto sort_checker = TEST_PREPARE_CALLABLE(std::ranges::sort);

    test_range_algo<0>{}(dpl_ranges::sort, sort_checker);
    test_range_algo<1>{}(dpl_ranges::sort, sort_checker, std::ranges::less{});

    test_range_algo<2>{}(dpl_ranges::sort, sort_checker, std::ranges::less{}, proj);
    test_range_algo<3>{}(dpl_ranges::sort, sort_checker, std::ranges::greater{}, proj);

    test_range_algo<4, P2>{}(dpl_ranges::sort, sort_checker, std::ranges::less{}, &P2::x);
    test_range_algo<5, P2>{}(dpl_ranges::sort, sort_checker, std::ranges::greater{}, &P2::x);

    test_range_algo<6, P2>{}(dpl_ranges::sort, sort_checker, std::ranges::less{}, &P2::proj);
    test_range_algo<7, P2>{}(dpl_ranges::sort, sort_checker, std::ranges::greater{}, &P2::proj);

    // Check larger specializations, use a custom comparator for a comparison-based sort (e.g. merge-sort)
    auto custom_less = [](const auto& a, const auto& b){ return a < b;};
    test_range_algo<8, std::uint16_t, data_in, DevicePolicy>{big_sz}(dpl_ranges::sort, sort_checker, custom_less);
    test_range_algo<9, int, data_in, DeviceAndParPolicies>{medium_sz}(dpl_ranges::sort, sort_checker, custom_less);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
