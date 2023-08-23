// -*- C++ -*-
//===-- oneapi_dpl_functional.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/functional>
#include "support/utils.h"

int main() {
    int a = 1, b = 2;
    oneapi::dpl::plus<int>{}(a, b);
    return TestUtils::done();
}
