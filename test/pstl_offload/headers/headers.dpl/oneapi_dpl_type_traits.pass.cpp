// -*- C++ -*-
//===-- oneapi_dpl_type_traits.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/type_traits>
#include "support/utils.h"

int main() {
    static_assert(oneapi::dpl::is_same<int, int>::value);
    return TestUtils::done();
}
