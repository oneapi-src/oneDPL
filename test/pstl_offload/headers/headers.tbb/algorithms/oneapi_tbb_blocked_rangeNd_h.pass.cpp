// -*- C++ -*-
//===-- oneapi_tbb_blocked_rangeNd_h.pass.cpp ----------------------------------===//
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

#define TBB_PREVIEW_BLOCKED_RANGE_ND 1
#include <oneapi/tbb/blocked_rangeNd.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::blocked_rangeNd<int, 1> br({0, 10});
    return TestUtils::done();
}
