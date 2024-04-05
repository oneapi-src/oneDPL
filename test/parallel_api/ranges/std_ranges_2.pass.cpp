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
#if 1
    test_range_algo{}(oneapi::dpl::ranges::count_if, std::ranges::count_if, pred, proj);
    test_range_algo{}(oneapi::dpl::ranges::count, std::ranges::count, 4, proj);

    test_range_algo<data_in_in>{}(oneapi::dpl::ranges::equal, std::ranges::equal, pred_2, proj, proj);

    test_range_algo{}(oneapi::dpl::ranges::is_sorted, std::ranges::is_sorted, std::ranges::less{}, proj);
    test_range_algo{}(oneapi::dpl::ranges::is_sorted, std::ranges::is_sorted, std::ranges::greater{}, proj);

    test_range_algo<data_in, /*RetTypeCheck*/true, /*ForwardRangeCheck*/false> sort_algo_test{};

    sort_algo_test(oneapi::dpl::ranges::sort, std::ranges::sort, std::ranges::less{}, proj);
    sort_algo_test(oneapi::dpl::ranges::sort, std::ranges::sort, std::ranges::greater{}, proj);
    sort_algo_test(oneapi::dpl::ranges::stable_sort, std::ranges::stable_sort, std::ranges::less{}, proj);
    sort_algo_test(oneapi::dpl::ranges::stable_sort, std::ranges::stable_sort, std::ranges::greater{}, proj);

    test_range_algo{}(oneapi::dpl::ranges::min_element, std::ranges::min_element, std::ranges::less{}, proj);
    test_range_algo{}(oneapi::dpl::ranges::min_element, std::ranges::min_element, std::ranges::greater{}, proj);

    test_range_algo{}(oneapi::dpl::ranges::max_element, std::ranges::max_element, std::ranges::less{}, proj);
    test_range_algo{}(oneapi::dpl::ranges::max_element, std::ranges::max_element, std::ranges::greater{}, proj);

    test_range_algo<data_in_out, /*RetTypeCheck*/false>{}(oneapi::dpl::ranges::copy,  std::ranges::copy);
    test_range_algo<data_in_out, /*RetTypeCheck*/false>{}(oneapi::dpl::ranges::copy_if,  std::ranges::copy_if,
        pred, proj);
#endif
    test_range_algo<data_in_in_out>{}(oneapi::dpl::ranges::merge, std::ranges::merge, std::ranges::less{}, proj, proj);
    test_range_algo<data_in_in_out>{}(oneapi::dpl::ranges::merge, std::ranges::merge, std::ranges::greater{}, proj, proj);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
