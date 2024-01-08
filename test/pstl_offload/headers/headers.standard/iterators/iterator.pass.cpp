// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include "support/utils.h"

int main() {
    int array[3];
    [[maybe_unused]] auto it = std::begin(array);
    return TestUtils::done();
}
