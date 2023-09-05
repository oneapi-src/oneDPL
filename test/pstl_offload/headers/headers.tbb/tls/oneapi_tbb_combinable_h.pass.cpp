// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/tbb/combinable.h>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::tbb::combinable<int> cb;
    return TestUtils::done();
}
