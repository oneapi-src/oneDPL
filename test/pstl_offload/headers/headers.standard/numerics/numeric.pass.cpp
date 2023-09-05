// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include "support/utils.h"

int main() {
    int array[2] = {0, 0};
    [[maybe_unused]] auto res = std::accumulate(array, array + 2, 0);
    return TestUtils::done();
}
