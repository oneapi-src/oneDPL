// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <optional>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::optional<int> opt;
    return TestUtils::done();
}
