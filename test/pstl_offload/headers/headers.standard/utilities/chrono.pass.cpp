// -*- C++ -*-
//===-- chrono.pass.cpp ----------------------------------===//
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

#include <chrono>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto tp1 = std::chrono::high_resolution_clock::now();
    return TestUtils::done();
}
