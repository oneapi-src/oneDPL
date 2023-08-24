// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/functional>
#include "support/utils.h"

int main() {
    int a = 1, b = 2;
    oneapi::dpl::plus<int>{}(a, b);
    return TestUtils::done();
}
