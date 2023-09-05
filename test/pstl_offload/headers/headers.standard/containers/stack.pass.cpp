// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stack>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::stack<int> s;
    return TestUtils::done();
}
