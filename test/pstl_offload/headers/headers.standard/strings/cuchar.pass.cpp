// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuchar>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::size_t r = sizeof(mbstate_t);
    return TestUtils::done();
}
