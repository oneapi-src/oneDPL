// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto are_same = std::is_same_v<int, float>;
    return TestUtils::done();
}
