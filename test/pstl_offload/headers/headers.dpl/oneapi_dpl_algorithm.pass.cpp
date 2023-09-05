// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>
#include "support/utils.h"

int main() {
    int array[3] = {3, 2, 1};
    oneapi::dpl::sort(oneapi::dpl::execution::seq, array, array + 3);
    return TestUtils::done();
}
