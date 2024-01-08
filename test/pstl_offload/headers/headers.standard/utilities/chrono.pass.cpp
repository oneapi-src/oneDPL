// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto tp1 = std::chrono::high_resolution_clock::now();
    return TestUtils::done();
}
