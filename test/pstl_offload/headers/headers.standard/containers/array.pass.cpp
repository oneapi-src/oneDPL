// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::array<int, 1> a;
    return TestUtils::done();
}
