// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define TBB_PREVIEW_MEMORY_POOL 1
#include <oneapi/tbb/memory_pool.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(oneapi::tbb::fixed_pool);
    return TestUtils::done();
}
