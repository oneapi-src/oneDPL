// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TBB ETS includes windows.h, that define max macro conflicting with max() from <limits>
// see https://github.com/oneapi-src/oneTBB/issues/832
#define NOMINMAX 1
#include <oneapi/tbb/combinable.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::combinable<int> cb;
    return TestUtils::done();
}
