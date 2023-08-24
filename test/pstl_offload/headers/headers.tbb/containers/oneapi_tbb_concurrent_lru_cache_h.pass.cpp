// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <oneapi/tbb/concurrent_lru_cache.h>
#include "support/utils.h"

float foo(int) { return 1.f; }

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_lru_cache<int, float> cache(&foo, 1);
    return TestUtils::done();
}
