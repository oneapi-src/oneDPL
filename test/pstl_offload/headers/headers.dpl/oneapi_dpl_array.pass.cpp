// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/array>
#include "support/utils.h"

int main() {
    [[maybe_unused]] oneapi::dpl::array<int, 1> arr = {1};
    return TestUtils::done();
}
