// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ranges>
#include "support/utils.h"

int main() {
    int array[1];
    // [[maybe_unused]] auto empty = std::ranges::empty(array);
    return TestUtils::done();
}
