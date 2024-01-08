// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <random>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto max = std::mt19937::max();
    return TestUtils::done();
}
