// -*- C++ -*-
//===-- limits.pass.cpp ----------------------------------===//
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

#include <limits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto is_signed = std::numeric_limits<size_t>::is_signed;
    return TestUtils::done();
}
