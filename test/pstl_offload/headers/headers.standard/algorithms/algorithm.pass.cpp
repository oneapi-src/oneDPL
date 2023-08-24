// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include "support/utils.h"

int main() {
    int array[3] = {3, 2, 1};
    std::sort(array, array + 3);
    return TestUtils::done();
}
