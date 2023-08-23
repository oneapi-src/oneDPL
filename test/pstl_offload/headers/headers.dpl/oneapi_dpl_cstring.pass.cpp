// -*- C++ -*-
//===-- oneapi_dpl_cstring.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/cstring>
#include "support/utils.h"

int main() {
    void* i = ::operator new(sizeof(int));
    oneapi::dpl::memset(i, 0, 1);
    ::operator delete(i);
    return TestUtils::done();
}
