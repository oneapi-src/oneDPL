// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <barrier>
#include "support/utils.h"

int main() {
    // [[maybe_unused]] auto m = std::barrier<>::max();
    return TestUtils::done();
}
