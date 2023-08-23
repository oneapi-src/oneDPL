// -*- C++ -*-
//===-- oneapi_dpl_cstddef.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/cstddef>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::dpl::size_t s = 0;
    return TestUtils::done();
}
