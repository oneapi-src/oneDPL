// -*- C++ -*-
//===-- oneapi_dpl_algorithm.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/algorithm>
#include "support/utils.h"

int main() {
    int array[3] = {3, 2, 1};
    oneapi::dpl::sort(oneapi::dpl::execution::seq, array, array + 3);
    return TestUtils::done();
}
