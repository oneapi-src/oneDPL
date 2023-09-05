// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <queue>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::queue<int> q;
    return TestUtils::done();
}
