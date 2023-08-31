// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/task_arena.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::task_arena arena;
    return TestUtils::done();
}
