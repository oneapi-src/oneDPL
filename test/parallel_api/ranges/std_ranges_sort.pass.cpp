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

    const std::size_t big_sz = 1<<25; //32M

    test_range_algo<0>{big_sz}(dpl_ranges::sort, std::ranges::sort);
    test_range_algo<1>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::less{});

    test_range_algo<2>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, proj);
    test_range_algo<3>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, proj);

    test_range_algo<4, P2>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, &P2::x);
    test_range_algo<5, P2>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, &P2::x);

    test_range_algo<6, P2>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::less{}, &P2::proj);
    test_range_algo<7, P2>{}(dpl_ranges::sort, std::ranges::sort, std::ranges::greater{}, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
