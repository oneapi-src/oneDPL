// -*- C++ -*-
//===-- oneapi_tbb_concurrent_priority_queue_h.pass.cpp ----------------------------------===//
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

#include <oneapi/tbb/concurrent_priority_queue.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_priority_queue<int> cpq;
    return TestUtils::done();
}
