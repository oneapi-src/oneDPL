// -*- C++ -*-
//===-- oneapi_dpl_execution.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/execution>
#include "support/utils.h"

int main() {
    static_assert(!std::is_same_v<decltype(oneapi::dpl::execution::par), void>);
    return TestUtils::done();
}
