// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <semaphore>
#include "support/utils.h"

int main() {
    // [[maybe_unused]] auto max = std::binary_semaphore::max();
    return TestUtils::done();
}
