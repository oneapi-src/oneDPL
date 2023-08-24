// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/utility>
#include "support/utils.h"

int main() {
    int a = 1;
    [[maybe_unused]] auto b = oneapi::dpl::move(a);
    return TestUtils::done();
}
