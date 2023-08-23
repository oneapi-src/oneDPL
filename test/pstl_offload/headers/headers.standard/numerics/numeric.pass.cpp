// -*- C++ -*-
//===-- numeric.pass.cpp ----------------------------------===//
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

#include <numeric>
#include "support/utils.h"

int main() {
    int array[2] = {0, 0};
    [[maybe_unused]] auto res = std::accumulate(array, array + 2, 0);
    return TestUtils::done();
}
