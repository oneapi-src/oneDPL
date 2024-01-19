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

#define _ONEDPL_ICPX_USE_KNOWN_IDENTITY_FOR_ARITHMETIC_64BIT_DATA_TYPES 0

#include "reduce_by_segment.h"

int
main()
{
    static_assert(_ONEDPL_ICPX_USE_KNOWN_IDENTITY_FOR_ARITHMETIC_64BIT_DATA_TYPES == 0);

    using ValueType = float;
    using BinaryPredicate = ::std::equal_to<ValueType>;
    using BinaryOperation = ::std::plus<ValueType>;
    run_test<ValueType, BinaryPredicate, BinaryOperation>();

    return TestUtils::done();
}
