// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cfloat>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto m = FLT_MIN;
    return TestUtils::done();
}
