// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <bitset>
#include "support/utils.h"

int main() {
    [[maybe_unused]] std::bitset<1> bt;
    return TestUtils::done();
}
