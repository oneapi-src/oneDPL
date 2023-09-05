// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <shared_mutex>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::shared_mutex m;
    return TestUtils::done();
}
