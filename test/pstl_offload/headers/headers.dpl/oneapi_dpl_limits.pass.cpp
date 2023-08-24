// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/limits>
#include "support/utils.h"

int main() {
    [[maybe_unused]] auto max = oneapi::dpl::numeric_limits<int>::max();
    return TestUtils::done();
}
