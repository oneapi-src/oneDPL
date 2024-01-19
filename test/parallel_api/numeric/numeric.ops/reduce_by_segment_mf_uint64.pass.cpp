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

#include "reduce_by_segment.h"

int
main()
{
    static_assert(_ONEDPL_ICPX_USE_KNOWN_IDENTITY_FOR_ARITHMETIC_64BIT_DATA_TYPES == 1);

    using ValueType = ::std::uint64_t;
    using BinaryPredicate = UserBinaryPredicate<ValueType>;
    using BinaryOperation = MaxFunctor<ValueType>;
    run_test<ValueType, BinaryPredicate, BinaryOperation>();

    return TestUtils::done();
}
