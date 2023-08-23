// -*- C++ -*-
//===-- type_traits.pass.cpp ----------------------------------===//
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

#include <type_traits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto are_same = std::is_same_v<int, float>;
    return TestUtils::done();
}
