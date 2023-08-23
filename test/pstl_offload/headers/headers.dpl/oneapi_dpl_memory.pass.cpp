// -*- C++ -*-
//===-- oneapi_dpl_memory.pass.cpp ----------------------------------===//
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

#include <oneapi/dpl/memory>
#include "support/utils.h"

int main() {
    int* i = reinterpret_cast<int*>(::operator new(sizeof(int)));
    oneapi::dpl::uninitialized_default_construct_n(oneapi::dpl::execution::seq, i, 1);
    ::operator delete(i);
    return TestUtils::done();
}
