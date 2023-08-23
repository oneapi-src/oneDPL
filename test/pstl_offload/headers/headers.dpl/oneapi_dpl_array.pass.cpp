// -*- C++ -*-
//===-- oneapi_dpl_array.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/array>
#include "support/utils.h"

int main() {
    oneapi::dpl::array<int, 1> array = {1};
    oneapi::dpl::get<0>(array) = 2;
    return TestUtils::done();
}
