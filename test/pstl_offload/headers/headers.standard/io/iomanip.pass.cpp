// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iomanip>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto r = std::setw(0);
    return TestUtils::done();
}
