// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <variant>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::variant<int, float> v = 1;
    return TestUtils::done();
}
