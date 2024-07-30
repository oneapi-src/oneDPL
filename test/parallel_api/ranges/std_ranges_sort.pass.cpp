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

    test_range_algo<int, data_in, /*RetTypeCheck*/true, /*ForwardRangeCheck*/false> sort_algo_test{};

    sort_algo_test(dpl_ranges::sort, std::ranges::sort, std::ranges::less{});
    sort_algo_test(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::less{});

    sort_algo_test(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, proj);
    sort_algo_test(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, proj);
    sort_algo_test(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::less{}, proj);
    sort_algo_test(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::greater{}, proj);

    test_range_algo<P2, data_in, /*RetTypeCheck*/true, /*ForwardRangeCheck*/false> sort_algo_test_m{};

    sort_algo_test_m(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, &P2::x);
    sort_algo_test_m(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, &P2::x);
    sort_algo_test_m(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::less{}, &P2::x);
    sort_algo_test_m(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::greater{}, &P2::x);

    sort_algo_test_m(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, &P2::proj);
    sort_algo_test_m(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, &P2::proj);
    sort_algo_test_m(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::less{}, &P2::proj);
    sort_algo_test_m(dpl_ranges::stable_sort, std::ranges::stable_sort, std::ranges::greater{}, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
