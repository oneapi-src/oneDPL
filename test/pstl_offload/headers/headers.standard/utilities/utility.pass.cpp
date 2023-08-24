// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>
#include "support/utils.h"

int main() {
    int a = 1, b = 2;
    std::swap(a, b);
    return TestUtils::done();
}
