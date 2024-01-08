// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <condition_variable>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::condition_variable cv;
    return TestUtils::done();
}
