// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define TBB_PREVIEW_BLOCKED_RANGE_ND 1
#include <oneapi/tbb/blocked_rangeNd.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::blocked_rangeNd<int, 1> br({0, 10});
    return TestUtils::done();
}
