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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include <iomanip>
#include <complex>

#if !TEST_ONLY_HETERO_POLICIES
void
foo()
{
    int* a = nullptr;
    double* b = nullptr;
    oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par_unseq, a, a, b, a, b);
}

void
bar()
{
    int* a = nullptr;
    std::complex<double>* b = nullptr;
    oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par_unseq, a, a, b, a, b);
}
#endif // !TEST_ONLY_HETERO_POLICIES

int main()
{
    int is_done = 0;
#if !TEST_ONLY_HETERO_POLICIES
    foo();
    bar();
    is_done = 1;
#endif // !TEST_ONLY_HETERO_POLICIES

    return TestUtils::done(is_done);
}
