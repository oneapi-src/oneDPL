// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::atomic<int> a = 0;
    return TestUtils::done();
}
