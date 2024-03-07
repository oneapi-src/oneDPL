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

#ifdef _PSTL_TEST_SET_INTERSECTION

    run_test_set<test_set_intersection>();
    bProcessed = true;

#endif // _PSTL_TEST_SET_INTERSECTION

    return TestUtils::done(bProcessed);
}
