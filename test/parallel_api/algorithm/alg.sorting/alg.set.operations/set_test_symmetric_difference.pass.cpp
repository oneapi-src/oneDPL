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

#include "set_common.h"

int
main()
{
    bool bProcessed = false;

    using data_t =
#if !ONEDPL_FPGA_DEVICE
        float64_t;
#else
        std::int32_t;
#endif

#ifdef _PSTL_TEST_SET_SYMMETRIC_DIFFERENCE

    test_set<test_set_symmetric_difference, data_t, data_t>(oneapi::dpl::__internal::__pstl_less(), false);
#if !ONEDPL_FPGA_DEVICE
    test_set<test_set_symmetric_difference, data_t, data_t>(oneapi::dpl::__internal::__pstl_less(), true);
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test_set<test_set_symmetric_difference, Num<std::int64_t>, Num<std::int32_t>>(
        [](const Num<std::int64_t>& x, const Num<std::int32_t>& y) { return x < y; }, true);

    test_set<test_set_symmetric_difference, MemoryChecker, MemoryChecker>(
        [](const MemoryChecker& val1, const MemoryChecker& val2) -> bool { return val1.value() < val2.value(); }, true);
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0,
                "wrong effect from set algorithms: number of ctor and dtor calls is not equal");
#endif

    bProcessed = true;

#endif // _PSTL_TEST_SET_SYMMETRIC_DIFFERENCE

    return TestUtils::done(bProcessed);
}
