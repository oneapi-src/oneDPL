// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unordered_set>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::unordered_set<int> us;
    return TestUtils::done();
}
