// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/tbb_allocator.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::tbb_allocator<int> alloc;
    return TestUtils::done();
}
