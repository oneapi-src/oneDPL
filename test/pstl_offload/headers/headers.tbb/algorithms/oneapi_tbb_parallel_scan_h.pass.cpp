// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/parallel_scan.h>
#include <oneapi/tbb/blocked_range.h>
#include "support/utils.h"

int main() {
    oneapi::tbb::blocked_range<int> range{0, 10};
    oneapi::tbb::parallel_scan(range, 0, [](const auto&, int a, bool) { return a; }, [](int a, int) { return a; });
    return TestUtils::done();
}
