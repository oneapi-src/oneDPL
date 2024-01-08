// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/blocked_range.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::blocked_range<int> br(0, 10);
    return TestUtils::done();
}
