// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cfenv>
#include "support/utils.h"

int main() {
    [[maybe_unused]] int i = std::feclearexcept(0);
    return TestUtils::done();
}
