// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto dummy = oneapi::dpl::execution::seq;
    return TestUtils::done();
}
