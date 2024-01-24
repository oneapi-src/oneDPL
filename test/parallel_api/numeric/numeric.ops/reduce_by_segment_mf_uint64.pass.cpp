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
#if defined(ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION)
    static_assert(false);
#endif

    using ValueType = ::std::uint64_t;
    using BinaryPredicate = UserBinaryPredicate<ValueType>;
    using BinaryOperation = MaxFunctor<ValueType>;
    run_test<ValueType, BinaryPredicate, BinaryOperation>();

    return TestUtils::done();
}
