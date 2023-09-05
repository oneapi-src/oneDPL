// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <thread>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(std::thread);
    return TestUtils::done();
}
