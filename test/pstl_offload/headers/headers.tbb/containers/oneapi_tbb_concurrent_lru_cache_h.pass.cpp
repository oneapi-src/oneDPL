// -*- C++ -*-
//===-- oneapi_tbb_concurrent_lru_cache_h.pass.cpp ----------------------------------===//
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

#define TBB_PREVIEW_CONCURRENT_LRU_CACHE 1
#include <oneapi/tbb/concurrent_lru_cache.h>
#include "support/utils.h"

float foo(int) { return 1.f; }

int main() {
    [[maybe_unused]] oneapi::tbb::concurrent_lru_cache<int, float> cache(&foo, 1);
    return TestUtils::done();
}
