// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/concurrent_map.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_map<int, float> cmap;
    return TestUtils::done();
}
