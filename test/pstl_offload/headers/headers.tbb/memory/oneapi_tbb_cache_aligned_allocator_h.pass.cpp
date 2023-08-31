// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/cache_aligned_allocator.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::cache_aligned_allocator<int> alloc;
    return TestUtils::done();
}
