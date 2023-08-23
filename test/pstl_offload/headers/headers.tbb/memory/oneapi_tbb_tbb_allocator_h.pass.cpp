// -*- C++ -*-
//===-- oneapi_tbb_tbb_allocator_h.pass.cpp ----------------------------------===//
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

#include <oneapi/tbb/tbb_allocator.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto is_equal = oneapi::tbb::tbb_allocator<int>::is_always_equal::value;
    return TestUtils::done();
}
