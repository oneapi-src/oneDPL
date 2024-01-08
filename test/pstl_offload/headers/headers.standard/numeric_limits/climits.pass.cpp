// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <climits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto bit = CHAR_BIT;
    return TestUtils::done();
}
