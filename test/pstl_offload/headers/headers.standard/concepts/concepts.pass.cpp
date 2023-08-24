// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <concepts>
#include "support/utils.h"

int main() {
    // [[maybe_unused]] bool b = std::same_as<int, float>;
    return TestUtils::done();
}
