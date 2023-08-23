// -*- C++ -*-
//===-- unordered_set.pass.cpp ----------------------------------===//
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

#include <unordered_set>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::unordered_set<int> us;
    return TestUtils::done();
}
