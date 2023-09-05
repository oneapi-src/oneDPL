// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cctype>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto res = std::tolower(1);
    return TestUtils::done();
}
