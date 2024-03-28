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

    test_range_algo{}(dpl_ranges::for_each, std::ranges::for_each, f_mutuable, proj_mutuable);

    //TODO: dpl_ranges::transform has output range as return type, std::ranges::trasnform - output iterator.
    test_range_algo<data_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, f, proj);

    test_range_algo{}(dpl_ranges::find_if, std::ranges::find_if, pred, proj);
    test_range_algo{}(dpl_ranges::find_if_not, std::ranges::find_if_not, pred, proj);
    test_range_algo{}(dpl_ranges::find, std::ranges::find, 4, proj);

    auto pred1 = [](auto&& val) -> decltype(auto) { return val > 0; };
    auto pred2 = [](auto&& val) -> decltype(auto) { return val == 4; };
    auto pred3 = [](auto&& val) -> decltype(auto) { return val < 0; };

    test_range_algo{}(dpl_ranges::all_of,  std::ranges::all_of, pred1, proj);
    test_range_algo{}(dpl_ranges::any_of,  std::ranges::any_of, pred2, std::identity{});
    test_range_algo{}(dpl_ranges::none_of,  std::ranges::none_of, pred3, std::identity{});

    test_range_algo{}(dpl_ranges::adjacent_find,  std::ranges::adjacent_find, pred_2, proj);

    test_range_algo<data_in_in>{}(dpl_ranges::search,  std::ranges::search, pred_2, proj);
    test_range_algo<data_in_val_n>{}(dpl_ranges::search_n,  std::ranges::search_n, pred_2, proj);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
