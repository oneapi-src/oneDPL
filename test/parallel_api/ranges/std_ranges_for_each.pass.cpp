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

    test_range_algo<int, data_in, false/*return type check*/>{}(dpl_ranges::for_each, std::ranges::for_each, f_mutuable);
    test_range_algo<int, data_in, false/*return type check*/>{}(dpl_ranges::for_each, std::ranges::for_each, f_mutuable, proj_mutuable);
    test_range_algo<P2, data_in, false/*return type check*/>{}(dpl_ranges::for_each, std::ranges::for_each, f_mutuable, &P2::x);
    test_range_algo<P2, data_in, false/*return type check*/>{}(dpl_ranges::for_each, std::ranges::for_each, f_mutuable, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
