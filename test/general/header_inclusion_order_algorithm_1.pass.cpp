// -*- C++ -*-
//===-- header_inclusion_order_algorithm_1.pass.cpp -----------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

int
main()
{

    return TestUtils::done();
}
