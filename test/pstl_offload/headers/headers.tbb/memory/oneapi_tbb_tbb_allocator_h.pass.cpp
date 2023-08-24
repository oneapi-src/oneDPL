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
    [[maybe_unused]] auto is_equal = oneapi::tbb::tbb_allocator<int>::is_always_equal::value;
    return TestUtils::done();
}
