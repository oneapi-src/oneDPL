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

#define ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION 1

#include "reduce_by_segment.h"

int
main()
{
    static_assert(ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION == 1);

    using ValueType = float;
    using BinaryPredicate = ::std::equal_to<ValueType>;
    using BinaryOperation = ::std::plus<ValueType>;
    run_test<ValueType, BinaryPredicate, BinaryOperation>();

    return TestUtils::done();
}
