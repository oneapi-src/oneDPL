// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <functional>
#include "support/utils.h"

int main() {
    [[maybe_unused]] bool are_equal = std::equal_to<>{}(1, 2);
    return TestUtils::done();
}
