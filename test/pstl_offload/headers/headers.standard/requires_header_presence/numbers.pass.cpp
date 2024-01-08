// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numbers>
#include "support/utils.h"

int main() {
#if __cpp_lib_math_constants >= 201907L
    [[maybe_unused]] auto phi = std::numbers::phi;
#endif
    return TestUtils::done();
}
