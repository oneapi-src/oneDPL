// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <deque>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::deque<int> d;
    return TestUtils::done();
}
