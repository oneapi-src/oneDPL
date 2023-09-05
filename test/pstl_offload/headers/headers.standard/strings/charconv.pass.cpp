// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <charconv>
#include "support/utils.h"

int main() {
    const char* str = "1";
    int n = 1;
    [[maybe_unused]] auto f = std::from_chars(str, str + 1, n);
    return TestUtils::done();
}
