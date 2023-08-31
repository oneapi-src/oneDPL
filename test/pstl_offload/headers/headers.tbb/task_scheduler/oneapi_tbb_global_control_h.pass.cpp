// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/global_control.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, 1);
    return TestUtils::done();
}
