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
#ifdef _PSTL_TEST_SET_DIFFERENCE
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const_set_difference<std::int32_t>>());
#endif

    return TestUtils::done(_PSTL_TEST_SET_DIFFERENCE);
}
