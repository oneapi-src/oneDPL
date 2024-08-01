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

    test_range_algo<0, int, data_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, f);
    test_range_algo<1, int, data_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, f, proj);
    test_range_algo<2, P2, data_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, f, &P2::x);
    test_range_algo<3, P2, data_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, f, &P2::proj);

    test_range_algo<4, int, data_in_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, binary_f);
    test_range_algo<5, int, data_in_in_out, false/*return type check*/>{}(dpl_ranges::transform, std::ranges::transform, binary_f, proj, proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
