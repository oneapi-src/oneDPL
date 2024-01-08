// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <memory>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::unique_ptr<int> uptr;
    return TestUtils::done();
}
